"""Audit complete matched QKVO training, restoration artifacts and quality/cost."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np,torch
from amp_recovery_state import lr_factor
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(ps):return hashlib.sha256(b''.join(x.numpy().tobytes() for x in ps.values())).hexdigest()
def main():
    name='matched-restore-training';a=R/f'exports/{name}-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/f'results/cloud-{name}-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        mn=name+'-manifest.json';entries=json.loads(t.extractfile(mn).read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{mn}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/f'provenance/{name}-protocol-v0.json';assert sha(pp)==sha(R/f'provenance/{name}-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    out=dest/f'results/{name}-stage-v0';control=load(out/'result.json');assert control['status']=='complete' and control['task_predictions']==1064 and control['optimizer_updates']==512 and control['gradient_passes']==8 and control['protocol_sha256']==sha(pp) and len(control['jobs'])==4
    train=np.load(dest/'data/32k-expanded-training-v0/train-calibration.npz')['train'];order=np.random.default_rng(2026091662).permutation(76).tolist()
    meta=load(dest/'data/fresh-word-dense-v0/tasks.json');families=sorted({x['family_id'] for x in meta if x['variant']=='long32768'});assert len(families)==32
    pg=np.load(dest/'data/pg19-external-v0/windows.npz')['windows'];by={};cost_by={};records=[];nll_count=0
    for j in p['jobs']:
        d=out/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==128 and v['task_predictions']==266 and v['nll_forwards']==126 and v['gradient_gate_passes']==2 and v['gradient_gate']['passed']
        assert v['eval_source_sha256']==sha(dest/'scripts/train_matched_restore_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu']
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        old=load(mirror/j['training_result']);assert old['identity']==v['identity']['parent'] and v['identity']['training_k']==j['training_k'] and v['identity']['protocol_sha256']==sha(pp) and v['identity']['trainable_parameters']==1081344
        assert v['identity']['trainable_projection_names']==['q_proj','k_proj','v_proj','o_proj']
        cal=next(e['values'] for e in old['evaluations'] if e['step']==0 and e['split']=='calibration');assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        source=torch.load(mirror/j['path'],map_location='cpu',weights_only=False)['params'];assert digest(source)==v['initial_parameter_digest']
        for step in [0,64,128]:
            f=d/f'checkpoint-{step}.pt';assert sha(f)==v['checkpoint_hashes'][str(step)];cp=torch.load(f,map_location='cpu',weights_only=False)
            assert cp['step']==cp['data_cursor']==step and cp['identity']==v['identity'] and set(cp['params'])==set(source) and len(cp['optimizer']['param_groups'])==1 and len(cp['optimizer']['param_groups'][0]['params'])==192
            if step==0:assert all(torch.equal(cp['params'][n],source[n]) for n in source)
            else:assert len(cp['optimizer']['state'])==192 and all(int(s['step'])==step for s in cp['optimizer']['state'].values())
        endpoint=cp['params'];assert any(not torch.equal(endpoint[n],source[n]) for n in source if '.q_proj.' in n or '.k_proj.' in n)
        restored=torch.load(d/'restored-adapter.pt',map_location='cpu',weights_only=False);abl=load(d/'ablation.json')
        assert abl==v['ablation'] and sha(d/'restored-adapter.pt')==abl['artifact_sha256'] and restored['source_step']==128 and restored['identity']==v['identity']
        expected={f'model.layers.{i}.self_attn.{q}.B' for i in range(24) for q in ['q_proj','k_proj']};assert set(abl['selected_B_tensors'])==expected and abl['selected_count']==48 and abl['unchanged_other_tensors']==144
        assert set(restored['params'])==set(endpoint) and all(torch.count_nonzero(restored['params'][n])==0 for n in expected) and all(torch.equal(restored['params'][n],endpoint[n]) for n in endpoint if n not in expected)
        rows=v['rows'];assert len(rows)==128 and rows==[json.loads(s) for s in (d/'training-steps.jsonl').read_text().splitlines()]
        for i,row in enumerate(rows):
            wi=order[i%76];assert row['step']==i+1 and row['window_index']==wi and row['input_sha256']==hashlib.sha256(train[wi].tobytes()).hexdigest() and row['training_k']==j['training_k'] and abs(row['lr']-.001*lr_factor(i,4,64))<1e-12
            assert all(np.isfinite(row[n]) for n in ['loss','gradient_norm','seconds']) and row['seconds']>0
        assert abs(v['complete_step_seconds']-sum(x['seconds'] for x in rows))<1e-8 and [x['step'] for x in v['evaluations']]==[64,128]
        assert v['loop_seconds']>=v['complete_step_seconds'] and v['training_completed_seconds']>=v['loop_seconds'] and abl['restore_seconds_including_serialization']>0
        assert abs(v['restored_artifact_seconds']-v['training_completed_seconds']-abl['restore_seconds_including_serialization'])<1e-8
        events=[json.loads(s) for s in (d/'events.jsonl').read_text().splitlines()];assert any(x['event']=='checkpoint_reload_verified' and x['step']==64 for x in events)
        flatrows=[json.loads(s) for s in (d/'task-predictions.jsonl').read_text().splitlines()];assert flatrows==[x for cc in v['conditions'] for x in cc['predictions']]
        assert [x['name'] for x in v['conditions']]==['original','restored']
        for condition in v['conditions']:
            cn=condition['name'];assert condition['parameter_digest']==digest(endpoint if cn=='original' else restored['params'])
            preds=condition['predictions'];assert len(preds)==133
            for x,m in zip(preds,meta):
                assert all(x[k]==value for k,value in m.items()) and x['condition']==cn and len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all() and x['prediction']==int(np.argmax(x['choice_logits']))
                if m['gold'] is not None:assert x['correct']==(x['prediction']==m['gold'])
            matrix=np.array([[next(x['correct'] for x in preds if x['family_id']==f and x['gold']==g) for g in range(4)] for f in families],dtype=float)
            short=sum(x['correct'] for x in preds if x['variant']=='short');assert len(condition['wiki_nll'])==9 and np.isfinite(condition['wiki_nll']).all()
            prows=condition['pg19'];assert len(prows)==16 and prows==[json.loads(s) for s in (d/(cn+'-pg19-nll.jsonl')).read_text().splitlines()]
            for i,(x,w) in enumerate(zip(prows,pg)):
                assert x['window']==i and x['input_sha256']==hashlib.sha256(w.tobytes()).hexdigest() and all(np.isfinite(x[m]) for m in ['full_nll','full_context_tail_nll','short_context_tail_nll','context_benefit_nats'])
                assert abs(x['context_benefit_nats']-(x['short_context_tail_nll']-x['full_context_tail_nll']))<1e-12
            by[(j['seed'],j['training_k'],cn)]=dict(matrix=matrix,short=short,wiki=np.array(condition['wiki_nll']),pg_full=np.array([x['full_nll'] for x in prows]),pg_tail=np.array([x['full_context_tail_nll'] for x in prows]))
            records.append(dict(seed=j['seed'],training_k=j['training_k'],condition=cn,word_accuracy=float(matrix.mean()),short_correct=short,wiki_ppl=float(np.exp(np.mean(condition['wiki_nll']))),pg19_ppl=float(np.exp(np.mean([x['full_nll'] for x in prows])))))
        cost_by[(j['seed'],j['training_k'])]=v;nll_count+=126
    assert nll_count==504;seeds=[2026091660,2026091661];rng=np.random.default_rng(p['primary']['bootstrap_seed']);contrasts=[]
    for label,ac,bc in [('primary','restored','original'),('symmetric_restored','restored','restored'),('original','original','original')]:
        for metric in ['matrix','wiki','pg_full','pg_tail']:
            ds=np.stack([by[(s,32,ac)][metric]-by[(s,0,bc)][metric] for s in seeds]);delta=ds.mean(axis=0)
            if metric=='matrix':delta=delta.mean(axis=1)
            ci=np.quantile(delta[rng.integers(0,len(delta),(p['primary']['draws'],len(delta)))].mean(axis=1),[.025,.975]);x=dict(contrast=label,metric=metric,difference=float(delta.mean()),ci95=ci.tolist())
            if metric!='matrix':x.update(ppl_ratio=float(np.exp(delta.mean())),ppl_ratio_95ci=np.exp(ci).tolist())
            contrasts.append(x)
    costs=[]
    for s in seeds:
        d=cost_by[(s,0)];sp=cost_by[(s,32)];costs.append(dict(seed=s,dense_process_seconds=d['training_completed_seconds'],sparse_restored_artifact_seconds=sp['restored_artifact_seconds'],artifact_saving_percent=100*(1-sp['restored_artifact_seconds']/d['training_completed_seconds']),step_saving_percent=100*(1-sp['complete_step_seconds']/d['complete_step_seconds']),loop_saving_percent=100*(1-sp['loop_seconds']/d['loop_seconds']),symmetric_restored_saving_percent=100*(1-sp['restored_artifact_seconds']/d['restored_artifact_seconds']),dense_restore_seconds=d['ablation']['restore_seconds_including_serialization'],sparse_restore_seconds=sp['ablation']['restore_seconds_including_serialization']))
    primary={x['metric']:x for x in contrasts if x['contrast']=='primary'}
    screen=dict(short=all(by[(s,k,c)]['short']==4 for s in seeds for k,c in [(0,'original'),(32,'restored')]),accuracy=primary['matrix']['ci95'][0]>-.05,wiki=primary['wiki']['ppl_ratio_95ci'][1]<=1.05,pg_full=primary['pg_full']['ppl_ratio_95ci'][1]<=1.05,pg_tail=primary['pg_tail']['ppl_ratio_95ci'][1]<=1.05,cost=all(x['artifact_saving_percent']>0 for x in costs) and float(np.median([x['artifact_saving_percent'] for x in costs]))>=5);screen['all']=all(screen.values())
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,scientific_updates=512,optimizer_updates=512,task_predictions=1064,nll_forwards=504,gradient_passes=8,records=records,cost=costs,contrasts=contrasts,replication_screen=screen,scope=p['scope']);target=R/f'results/{name}-audit-v0';target.mkdir(exist_ok=True);(target/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 完整匹配训练与QK恢复：质量—成本复验','','同一4090，两种子从初始检查点各做128步密集/K32 QKVO LoRA训练。统一非重入梯度检查点、数据顺序和优化器；这是原有方法的同卡实际成本复验，已用过的评测数据不是新独立测试。','','|种子|K|评测权重|长题正确率|Wiki困惑度|PG19子集困惑度|','|---|---:|---|---:|---:|---:|']
    for x in records:lines.append(f"|{x['seed']}|{x['training_k']}|{x['condition']}|{x['word_accuracy']*100:.2f}%|{x['wiki_ppl']:.4f}|{x['pg19_ppl']:.4f}|")
    lines+=['','成本：'+json.dumps(costs,ensure_ascii=False),'','预定复验筛查：'+json.dumps(screen,ensure_ascii=False),'','主成本为完整进程到训练完成的时间，加上稀疏模型恢复及适配器写盘时间；双方均不包含最后科研质量评测。训练过程的校准、检查点、预检与初始化计入。为比较恢复前后而保存原权重的诊断性CPU副本不计入部署恢复成本。另报训练步、循环、对称恢复成本。不是完整基座预训练、官方PG19榜单或成本收敛曲线；QK-Restore已有前作。所有原始和恢复后检查点、步日志、逐题/逐书结果、失败、UTC与审计均保存。']
    (R/'docs/matched-restore-training-results-2026-09-17.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',screen=screen,cost=costs)))
if __name__=='__main__':main()
