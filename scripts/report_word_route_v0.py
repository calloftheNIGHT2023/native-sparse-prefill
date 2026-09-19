"""Verify archived routing diagnostic and report every preregistered contrast."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/word-route-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-word-route-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('word-route-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'word-route-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/word-route-protocol-v0.json';assert sha(pp)==sha(R/'provenance/word-route-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    assert sha(R/'results/word-operator-swap-audit-v0/result.json')==p['parent_audit_sha256']
    stage=dest/'results/word-route-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and control['optimizer_updates']==0 and control['task_predictions']==384 and control['protocol_sha256']==sha(pp)
    assert len(control['jobs'])==len(p['jobs'])==6 and control['seconds']<=p['maximum_seconds']
    meta=[x for x in load(dest/'data/word-counterfactual-v0/tasks.json') if x['variant']=='long32768'];placements=load(dest/'data/word-route-v0/placements.json')
    arrays={};seedrows=[];environment=None
    for j,run in zip(p['jobs'],control['jobs']):
        assert run['name']==j['name'] and run['returncode']==0 and run['status']=='complete'
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==0 and v['step']==128 and v['task_predictions']==64
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_word_route_v0.py')==sha(d/'source.py')
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        train=load(mirror/j['training_result']);assert v['identity']==train['identity'] and v['environment']['gpu']==p['evaluation_gpu']
        if environment is None:environment=v['environment']
        else:assert environment==v['environment']
        cal=next(e['values'] for e in train['evaluations'] if e['step']==128 and e['split']=='calibration');assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        preds=v['predictions'];assert len(preds)==64 and preds==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()]
        reference=load(dest/j['reference']);assert sha(dest/j['reference'])==j['reference_sha256'] and reference['environment']==environment
        ref={x['item_id']:x for x in reference['predictions'] if x['variant']=='long32768'}
        for x,y in zip(preds,meta):
            assert all(x[n]==value for n,value in y.items()) and len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all()
            assert x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold']) and x['placement']==placements[x['item_id']]
            assert len(x['route_telemetry'])==24 and all(0<=z['target_block_recall_at_final_query']<=1 and z['target_blocks']==1 for z in x['route_telemetry'])
            if j['condition']=='baseline':
                old=ref[x['item_id']];assert max(abs(a-b) for a,b in zip(x['choice_logits'],old['choice_logits']))<=1e-6 and x['prediction']==old['prediction']
        if j['condition']=='baseline':assert v['task_replay_max_abs_error']<=1e-6
        else:assert v['task_replay_max_abs_error'] is None
        values=np.array([x['correct'] for x in preds],dtype=float).reshape(32,2);arrays[(j['condition'],j['seed'])]=values
        seedrows.append(dict(condition=j['condition'],seed=j['seed'],accuracy=float(values.mean()),both_correct=float(np.all(values==1,axis=1).mean()),route_coverage_before_intervention=float(np.mean([z['target_block_recall_at_final_query'] for x in preds for z in x['route_telemetry']]))))
    rng=np.random.default_rng(p['bootstrap']['seed']);draws=rng.integers(0,32,size=(p['bootstrap']['draws'],32))
    points={};summary=[]
    for condition in ['baseline','target','sham']:
        selected=[arrays[(condition,seed)] for seed in p['seeds']];points[condition]=np.mean([x.mean(axis=1) for x in selected],axis=0)
        summary.append(dict(condition=condition,accuracy=float(points[condition].mean()),both_correct=float(np.mean([np.all(x==1,axis=1).mean() for x in selected]))))
    contrasts={}
    for name in p['primary_contrasts']:
        a,b=name.split('-');diff=points[a]-points[b];contrasts[name]=dict(mean_pp=float(diff.mean()*100),ci95_pp=(np.quantile(diff[draws].mean(axis=1),[.025,.975])*100).tolist())
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,optimizer_updates=0,task_predictions=384,summary=summary,seed_rows=seedrows,contrasts=contrasts,control=control,bootstrap=p['bootstrap'],scope=p['scope'])
    out=R/'results/word-route-audit-v0';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 事实词：固定稀疏预算的已知位置干预','','固定128步K32权重、两个种子、32组背景；只在问题后缀处强制加入值词所在块或预定无关块，每头仍32块。两组原生基线的128条长文本预测先全部通过1e-6重放，再做干预。新增384次计分前向，0训练更新。','','|条件|准确率|事实两版都正确率|','|---|---:|---:|']
    for x in summary:lines.append(f"|{x['condition']}|{x['accuracy']:.2%}|{x['both_correct']:.2%}|")
    lines+=['','|配对对照|变化pp|背景组95%区间pp|','|---|---:|---|']
    for name,e in contrasts.items():
        lo,hi=e['ci95_pp'];lines.append(f"|{name}|{e['mean_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
    lines+=['','这是知道答案位置的机制诊断，不能当作可部署方法、独立验证或加速证据；不保证强制包含事实句所有前缀词。背景组共同重采样保留事实对及两个种子。无关块也可能含有有用信息；对照用于检验相同操作的影响，不声称无关块绝对无用。保留此前RACE任务上的强制路由失败，不将新结果泛化到所有任务。','','|作业|开始UTC|结束UTC|','|---|---|---|']
    for x in control['jobs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|")
    (R/'docs/word-route-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',summary=summary,contrasts=contrasts)))
if __name__=='__main__':main()
