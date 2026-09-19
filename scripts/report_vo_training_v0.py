"""Verify frozen-QK actual-training traces, checkpoints, quality and matched full-step costs."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np,torch
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/vo-training-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-vo-training-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('vo-training-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'vo-training-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/vo-training-protocol-v0.json';assert sha(pp)==sha(R/'provenance/vo-training-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    stage=dest/'results/vo-training-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and control['task_predictions']==768 and control['optimizer_updates']==512 and control['gradient_passes']==8 and control['protocol_sha256']==sha(pp)
    assert sha(R/'results/qk-restore-baseline-audit-v0/result.json')==p['parent_audit_sha256']
    train=np.load(dest/'data/32k-expanded-training-v0/train-calibration.npz')['train'];order=np.random.default_rng(2026091662).permutation(76).tolist();meta=load(dest/'data/word-counterfactual-v0/tasks.json');families=sorted({x['family_id'] for x in meta});records=[];by={}
    from amp_recovery_state import lr_factor
    for j in p['jobs']:
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==128 and v['task_predictions']==192 and v['step']==128 and v['gradient_gate_passes']==2 and v['gradient_gate']['passed']
        assert v['eval_source_sha256']==sha(dest/'scripts/train_vo_only_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu'] and v['frozen_qk_unchanged']
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        old=load(mirror/j['training_result']);assert old['identity']==v['identity']['parent'] and v['identity']['training_k']==j['training_k'] and v['identity']['protocol_sha256']==sha(pp) and v['identity']['trainable_parameters']==540672
        cal=next(e['values'] for e in old['evaluations'] if e['step']==0 and e['split']=='calibration');assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        source=torch.load(mirror/j['path'],map_location='cpu',weights_only=False)['params'];qk=[n for n in source if any('.'+q+'.' in n for q in ['q_proj','k_proj'])];assert len(qk)==96
        for step in [0,64,128]:
            cp=d/f'checkpoint-{step}.pt';assert sha(cp)==v['checkpoint_hashes'][str(step)];x=torch.load(cp,map_location='cpu',weights_only=False)
            assert x['step']==x['data_cursor']==step and x['identity']==v['identity'] and set(x['params'])==set(source)
            assert all(torch.equal(x['params'][n],source[n]) for n in qk)
            if step==0:assert all(torch.equal(x['params'][n],source[n]) for n in source)
            assert len(x['optimizer']['param_groups'])==1 and len(x['optimizer']['param_groups'][0]['params'])==96
        digest=hashlib.sha256(b''.join(source[n].numpy().tobytes() for n in qk)).hexdigest();assert digest==v['frozen_qk_digest'] and v['initial_vo_digest']!=v['final_vo_digest']
        rows=v['rows'];assert len(rows)==128 and rows==[json.loads(s) for s in (d/'training-steps.jsonl').read_text().splitlines()]
        for i,row in enumerate(rows):
            wi=order[i%76];assert row['step']==i+1 and row['window_index']==wi and row['input_sha256']==hashlib.sha256(train[wi].tobytes()).hexdigest() and row['training_k']==j['training_k'] and abs(row['lr']-.001*lr_factor(i,4,64))<1e-12
            assert all(np.isfinite(row[n]) for n in ['loss','gradient_norm','seconds']) and row['seconds']>0
        assert abs(v['complete_step_seconds']-sum(x['seconds'] for x in rows))<1e-8 and [x['step'] for x in v['evaluations']]==[64,128]
        events=[json.loads(s) for s in (d/'events.jsonl').read_text().splitlines()];assert any(x['event']=='checkpoint_reload_verified' and x['step']==64 for x in events)
        preds=v['predictions'];assert len(preds)==192 and preds==[json.loads(s) for s in (d/'task-predictions.jsonl').read_text().splitlines()]
        for x,m in zip(preds,meta):assert all(x[k]==value for k,value in m.items()) and np.isfinite(x['choice_logits']).all() and x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
        matrix=np.array([[next(x['correct'] for x in preds if x['family_id']==f and x['counterfactual']==cf and x['variant']=='long32768') for cf in [0,1]] for f in families],dtype=float)
        assert len(v['report_nll'])==9 and np.isfinite(v['report_nll']).all();v['matrix']=matrix;by[(j['seed'],j['training_k'])]=v
        records.append(dict(seed=j['seed'],training_k=j['training_k'],complete_step_seconds=v['complete_step_seconds'],loop_seconds=v['loop_seconds'],training_completed_seconds=v['training_completed_seconds'],long_accuracy=float(matrix.mean()),short_correct=sum(x['correct'] for x in preds if x['variant']=='short'),mean_nll=float(np.mean(v['report_nll'])),perplexity=float(np.exp(np.mean(v['report_nll']))),peak_gib=v['peak_allocated_bytes']/2**30))
    assert len(records)==4;diff=[];nd=[];cost=[]
    for seed in [2026091660,2026091661]:
        d=by[(seed,0)];s=by[(seed,32)];diff.append((s['matrix']-d['matrix']).mean(axis=1));nd.append(np.array(s['report_nll'])-d['report_nll']);cost.append(dict(seed=seed,step_saving_percent=100*(1-s['complete_step_seconds']/d['complete_step_seconds']),loop_saving_percent=100*(1-s['loop_seconds']/d['loop_seconds']),training_process_saving_percent=100*(1-s['training_completed_seconds']/d['training_completed_seconds'])))
    rng=np.random.default_rng(2026091675);delta=np.mean(diff,axis=0);ci=np.quantile(delta[rng.integers(0,32,(20000,32))].mean(axis=1)*100,[.025,.975]).tolist();nll=np.mean(nd,axis=0);nc=np.quantile(nll[rng.integers(0,9,(20000,9))].mean(axis=1),[.025,.975])
    quality=dict(accuracy_difference_pp=float(delta.mean()*100),background_paired_95ci_pp=ci,perplexity_ratio=float(np.exp(nll.mean())),conditional_window_perplexity_ratio_95ci=np.exp(nc).tolist())
    screen=dict(accuracy=ci[0]>-5,perplexity=float(np.exp(nc[1]))<=1.05,cost=float(np.median([x['step_saving_percent'] for x in cost]))>=5 and all(x['step_saving_percent']>0 for x in cost));screen['all']=all(screen.values())
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,optimizer_updates=512,scientific_updates=512,task_predictions=768,gradient_passes=8,records=records,cost=cost,quality=quality,development_screen=screen,scope=p['scope']);out=R/'results/vo-training-audit-v0';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 从头固定Q/K的V/O继续训练开发实验','','两个种子各进行128步密集或K32训练，Q/K保持训练前数值不变；其他基础权重冻结，只训练540672个V/O LoRA参数。所有评测使用密集注意力。512次科学更新；不是从头全参数预训练。旧数据仅用于开发筛查。','','|种子|训练K|128步秒|含检查点/校准循环秒|长题正确率|困惑度|峰值GiB|','|---|---:|---:|---:|---:|---:|---:|']
    for x in records:lines.append(f"|{x['seed']}|{x['training_k']}|{x['complete_step_seconds']:.2f}|{x['loop_seconds']:.2f}|{x['long_accuracy']*100:.2f}%|{x['perplexity']:.4f}|{x['peak_gib']:.2f}|")
    lines+=['','质量对照：'+json.dumps(quality,ensure_ascii=False),'','成本对照：'+json.dumps(cost,ensure_ascii=False),'','预先固定的开发筛查：'+json.dumps(screen,ensure_ascii=False),'','计时包括输入创建/搬运、完整前反向与梯度检查点重算、裁剪、AdamW和学习率更新。循环另含64/128步密集校准、检查点写盘及64步重载。进程训练完成时间另含模型加载和数值预检。最终任务评测不计入训练完成时间。统计区间以旧背景/旧文本窗口为单位，两个种子；不能视为独立泛化确认。固定Q/K是常见控制策略，邻近前作QK-Restore和SSA限制原创性主张。']
    (R/'docs/vo-training-results-2026-09-17.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',quality=quality,cost=cost,screen=screen)))
if __name__=='__main__':main()
