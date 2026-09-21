"""Verify known QK-Restore control and its interaction with sparse training on development data."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/qk-restore-baseline-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-qk-restore-baseline-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('qk-restore-baseline-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'qk-restore-baseline-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/qk-restore-baseline-protocol-v0.json';assert sha(pp)==sha(R/'provenance/qk-restore-baseline-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    stage=dest/'results/qk-restore-baseline-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and control['task_predictions']==768 and control['optimizer_updates']==control['gradient_passes']==0 and control['protocol_sha256']==sha(pp)
    meta=load(dest/'data/word-counterfactual-v0/tasks.json');families=sorted({x['family_id'] for x in meta});assert len(families)==32
    records=[];changes={0:[],32:[]};ratios={0:[],32:[]}
    for j in p['jobs']:
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==0 and v['task_predictions']==192 and v['evaluation_k']==0
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_qk_restore_baseline_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu']
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        old=load(mirror/j['training_result']);assert old['identity']==v['identity'];cal=next(e['values'] for e in old['evaluations'] if e['step']==128 and e['split']=='calibration');assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        ab=v['ablation'];assert ab==load(d/'ablation.json') and ab['selected_count']==48 and ab['unchanged_other_tensors']==144 and ab['restored_to_pretraining_QK'] and len(set(ab['selected_B_tensors']))==48
        assert all(n.endswith('.B') and any('.'+q+'.' in n for q in ['q_proj','k_proj']) for n in ab['selected_B_tensors'])
        rows=v['predictions'];assert len(rows)==192 and rows==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()]
        for x,m in zip(rows,meta):assert all(x[k]==value for k,value in m.items()) and np.isfinite(x['choice_logits']).all() and x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
        wordref=next(z for z in p['references'] if z['kind']=='word' and z['training_k']==j['k'] and z['seed']==j['seed']);base=load(dest/wordref['path'])['predictions']
        def matrix(z):return np.array([[next(x['correct'] for x in z if x['family_id']==family and x['counterfactual']==cf and x['variant']=='long32768') for cf in [0,1]] for family in families],dtype=float)
        bm=matrix(base);nm=matrix(rows);changes[j['k']].append((nm-bm).mean(axis=1))
        nr=next(z for z in p['references'] if z['kind']=='nll' and z['training_k']==j['k'] and z['seed']==j['seed']);bn=np.array([x['full_nll'] for x in load(dest/nr['path'])['windows']]);nn=np.array(v['report_nll']);assert len(nn)==9 and np.isfinite(nn).all() and v['report_nll']==load(d/'report-nll.json')
        ratios[j['k']].append(nn-bn)
        records.append(dict(k=j['k'],seed=j['seed'],original_accuracy=float(bm.mean()),restored_accuracy=float(nm.mean()),restored_short_correct=sum(x['correct'] for x in rows if x['variant']=='short'),original_nll=float(bn.mean()),restored_nll=float(nn.mean()),perplexity_ratio_vs_original=float(np.exp((nn-bn).mean()))))
    assert len(records)==4
    idx=np.random.default_rng(2026091674).integers(0,32,(20000,32));effects=[]
    for k in [0,32]:
        delta=np.mean(changes[k],axis=0);effects.append(dict(training_k=k,accuracy_change_pp=float(delta.mean()*100),background_paired_95ci_pp=np.quantile(delta[idx].mean(axis=1)*100,[.025,.975]).tolist(),mean_nll_change=float(np.mean(ratios[k]))))
    interaction=np.mean(changes[32],axis=0)-np.mean(changes[0],axis=0)
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,optimizer_updates=0,task_predictions=768,records=records,effects=effects,interaction_pp=float(interaction.mean()*100),interaction_95ci_pp=np.quantile(interaction[idx].mean(axis=1)*100,[.025,.975]).tolist(),scope=p['scope']);out=R/'results/qk-restore-baseline-audit-v0';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 已有QK-Restore基线：训练损失定位','','恢复Q/K为已有方法，见https://arxiv.org/abs/2606.11052。此轮只用于对照诊断，不是原创方法或独立确认。0优化器更新，768任务前向，旧32背景开发集及9个旧WikiText窗口。所有模型使用密集评测；只将48个Q/K LoRA B矩阵置零，其余144个适配器张量逐位不变。','','|训练K|种子|原长题正确率|恢复QK后|原NLL|恢复QK后NLL|','|---|---:|---:|---:|---:|---:|']
    for x in records:lines.append(f"|{x['k']}|{x['seed']}|{x['original_accuracy']*100:.2f}%|{x['restored_accuracy']*100:.2f}%|{x['original_nll']:.6f}|{x['restored_nll']:.6f}|")
    for x in effects:lines+=['',f"训练K{x['training_k']}恢复QK的准确率变化{x['accuracy_change_pp']:+.2f}pp，背景配对95%区间{x['background_paired_95ci_pp']}；NLL变化{x['mean_nll_change']:+.6f}。"]
    lines+=['',f"稀疏组恢复收益减密集组恢复收益：{result['interaction_pp']:+.2f}pp，95%区间{result['interaction_95ci_pp']}。",'','事后参数恢复的结果不能证明训练时冻结Q/K会产生相同效果。若两组都出现类似现象，则不能据此主张稀疏特有机制。真实训练节省与达到同等质量所需总成本仍需独立对照。']
    (R/'docs/qk-restore-baseline-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',effects=effects,interaction_pp=result['interaction_pp'])))
if __name__=='__main__':main()
