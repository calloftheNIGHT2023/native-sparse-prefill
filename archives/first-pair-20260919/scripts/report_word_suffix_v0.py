"""Audit answer-location-blind suffix intervention against all fixed references."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/word-suffix-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-word-suffix-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('word-suffix-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'word-suffix-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/word-suffix-protocol-v0.json';assert sha(pp)==sha(R/'provenance/word-suffix-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    assert sha(R/'results/word-route-audit-v0/result.json')==p['parent_audit_sha256']
    stage=dest/'results/word-suffix-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and control['optimizer_updates']==0 and control['task_predictions']==256 and control['protocol_sha256']==sha(pp)
    assert len(control['jobs'])==len(p['jobs'])==4 and control['seconds']<=p['maximum_seconds']
    meta=[x for x in load(dest/'data/word-counterfactual-v0/tasks.json') if x['variant']=='long32768'];arrays={};seedrows=[];environment=None
    def add(v,training_k,condition,seed):
        nonlocal environment
        if environment is None:environment=v['environment']
        else:assert environment==v['environment']
        assert v['environment']['gpu']==p['evaluation_gpu']
        preds=[x for x in v['predictions'] if x['variant']=='long32768'];assert len(preds)==64
        for x,y in zip(preds,meta):
            assert all(x[n]==value for n,value in y.items()) and len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all()
            assert x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
            if condition=='SD':assert x['suffix_start']==32768-len(p['suffix_token_ids'])
        values=np.array([x['correct'] for x in preds],dtype=float).reshape(32,2);arrays[(training_k,condition,seed)]=values
        seedrows.append(dict(training_k=training_k,condition=condition,seed=seed,accuracy=float(values.mean()),both_correct=float(np.all(values==1,axis=1).mean())))
    for ref in p['references']:
        f=dest/ref['path'];assert sha(f)==ref['sha256'];v=load(f);j=v['job'];assert j['k']==ref['training_k'] and j['seed']==ref['seed'] and j['step']==128 and v.get('evaluation_k',j['k'])==ref['evaluation_k']
        add(v,ref['training_k'],'DD' if ref['evaluation_k']==0 else 'SS',ref['seed'])
    for j,run in zip(p['jobs'],control['jobs']):
        assert run['name']==j['name'] and run['returncode']==0 and run['status']=='complete'
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==0 and v['step']==128 and v['task_predictions']==64
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_word_suffix_v0.py')==sha(d/'source.py')
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        train=load(mirror/j['training_result']);assert v['identity']==train['identity']
        cal=next(e['values'] for e in train['evaluations'] if e['step']==128 and e['split']=='calibration');assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        assert v['predictions']==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()];add(v,j['k'],'SD',j['seed'])
    assert len(arrays)==12
    rng=np.random.default_rng(p['bootstrap']['seed']);draws=rng.integers(0,32,size=(p['bootstrap']['draws'],32));points={};summary=[];contrasts={}
    for k in [0,32]:
        for condition in ['SS','SD','DD']:
            selected=[arrays[(k,condition,seed)] for seed in p['seeds']];points[k,condition]=np.mean([x.mean(axis=1) for x in selected],axis=0)
            summary.append(dict(training_k=k,condition=condition,accuracy=float(points[k,condition].mean()),both_correct=float(np.mean([np.all(x==1,axis=1).mean() for x in selected]))))
        for left,right in [('SD','SS'),('SD','DD')]:
            diff=points[k,left]-points[k,right];contrasts[f'train{k}:{left}-{right}']=dict(mean_pp=float(diff.mean()*100),ci95_pp=(np.quantile(diff[draws].mean(axis=1),[.025,.975])*100).tolist())
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,optimizer_updates=0,task_predictions=256,reused_long_predictions=512,summary=summary,seed_rows=seedrows,contrasts=contrasts,control=control,bootstrap=p['bootstrap'],scope=p['scope'])
    out=R/'results/word-suffix-audit-v0';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 事实词：不使用答案位置的密集问题后缀','','固定128步、密集/K32训练、两个种子、32组背景。SS为全稀疏，SD为稀疏前缀+密集问题后缀，DD为全密集。只新增4个SD作业，共256次前向，0训练。所有模型先按原生注意力重放校准NLL（1e-6）。','','|训练|评测模式|准确率|事实两版都正确率|','|---|---|---:|---:|']
    for x in summary:lines.append(f"|K{x['training_k']}|{x['condition']}|{x['accuracy']:.2%}|{x['both_correct']:.2%}|")
    lines+=['','|配对对照|变化pp|背景组95%区间pp|','|---|---:|---|']
    for name,e in contrasts.items():
        lo,hi=e['ci95_pp'];lines.append(f"|{name}|{e['mean_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
    lines+=['','这是已知后缀保护思想的诊断基线，不主张原创方法。后缀边界来自固定问题模板，不提供答案词或答案位置给注意力算子；预检已验证右下对齐因果约束。数据已用于开发，不是独立确认集。计时包含先算后丢弃的稀疏后缀，不能从交互数量推断端到端加速。不把未显著低于DD解释为质量等价。原v0预检因缺失CUBLAS确定性环境失败，v1仅修正环境后通过，日志保留。','','|作业|开始UTC|结束UTC|','|---|---|---|']
    for x in control['jobs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|")
    (R/'docs/word-suffix-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',summary=summary,contrasts=contrasts)))
if __name__=='__main__':main()
