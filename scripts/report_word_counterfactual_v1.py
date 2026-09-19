"""Verify full fixed trajectories on the frozen assigned-word task."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/word-counterfactual-evidence-v1.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-word-counterfactual-evidence-v1';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('word-counterfactual-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'word-counterfactual-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/word-counterfactual-protocol-v1.json';assert sha(pp)==sha(R/'provenance/word-counterfactual-protocol-v1.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    gate=R/'results/word-counterfactual-audit-v0/result.json';assert sha(gate)==p['parent_gate_audit_sha256'] and load(gate)['status']=='verified'
    reference=dest/p['reference'];assert sha(reference)==p['reference_sha256']==sha(R/'results/cloud-word-counterfactual-evidence-v0'/p['reference']);base=load(reference)
    stage=dest/'results/word-counterfactual-stage-v1';control=load(stage/'result.json');assert control['status']=='complete' and control['optimizer_updates']==0 and control['task_predictions']==1728 and control['protocol_sha256']==sha(pp)
    assert len(control['jobs'])==len(p['jobs'])==9
    meta=load(dest/'data/word-counterfactual-v0/tasks.json');assert len(meta)==192
    arrays={};seedrows=[]
    def add(v):
        preds=v['predictions'];assert len(preds)==v['task_predictions']==192
        for x,y in zip(preds,meta):
            assert all(x[n]==value for n,value in y.items()) and len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all()
            assert x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
        by={variant:np.array([x['correct'] for x in preds if x['variant']==variant],dtype=float).reshape(32,2) for variant in ['short','no_context','long32768']}
        for variant in ['short','long32768']:
            values=by[variant];record=v['short_gate' if variant=='short' else 'long_gate'];criteria=p['short_gate' if variant=='short' else 'long_gate']
            expected=dict(correct=int(values.sum()),total=64,pairs_both_correct=int(np.all(values==1,axis=1).sum()),pairs_total=32,gain_over_no_context=int(values.sum()-by['no_context'].sum()))
            expected['passed']=expected['correct']>=criteria['min_correct'] and expected['pairs_both_correct']>=criteria['min_pairs_both_correct'] and expected['gain_over_no_context']>=criteria['min_gain_over_no_context'];assert record==expected
        j=v['job'];arrays[(j['k'],j['step'],j['seed'])]=by
        seedrows.append(dict(k=j['k'],step=j['step'],seed=j['seed'],short_accuracy=float(by['short'].mean()),no_context_accuracy=float(by['no_context'].mean()),long_accuracy=float(by['long32768'].mean()),long_both_correct=float(np.all(by['long32768']==1,axis=1).mean())))
    add(base)
    for j,run in zip(p['jobs'],control['jobs']):
        assert run['name']==j['name'] and run['returncode']==0 and run['status']=='complete'
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==0 and v['step']==j['step']
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_word_counterfactual_v1.py')==sha(d/'source.py')
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        train=load(mirror/j['training_result']);assert v['identity']==train['identity'] and v['environment']==base['environment']
        cal=next(e['values'] for e in train['evaluations'] if e['step']==j['step'] and e['split']=='calibration');assert max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        assert v['predictions']==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()];add(v)
    assert len(arrays)==10
    summary=[];contrasts=[];rng=np.random.default_rng(2026091694);draws=rng.integers(0,32,size=(10000,32))
    def estimate(a):return dict(mean_pp=float(a.mean()*100),ci95_pp=(np.quantile(a[draws].mean(axis=1),[.025,.975])*100).tolist())
    for step in [0,64,128]:
        points={}
        for k in [0,32]:
            selected=[v for (kk,ss,_),v in arrays.items() if kk==k and ss==step];assert len(selected)==(1 if step==0 else 2)
            mean=np.mean([v['long32768'].mean(axis=1) for v in selected],axis=0);both=np.mean([np.all(v['long32768']==1,axis=1) for v in selected],axis=0);points[k]=(mean,both)
            summary.append(dict(k=k,step=step,short_accuracy=float(np.mean([v['short'].mean() for v in selected])),no_context_accuracy=float(np.mean([v['no_context'].mean() for v in selected])),long_accuracy=float(mean.mean()),long_both_correct=float(both.mean())))
        contrasts.append(dict(step=step,long_sparse_minus_dense=estimate(points[32][0]-points[0][0]),both_correct_sparse_minus_dense=estimate(points[32][1]-points[0][1])))
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,optimizer_updates=0,task_predictions=1728,reused_base_predictions=192,summary=summary,seed_rows=seedrows,contrasts=contrasts,control=control,bootstrap=dict(unit='background_family',n=32,draws=10000,seed=2026091694,scope='Conditional exploratory intervals jointly preserving factualpairs andseeds; short/no-context repeats are notindependent questions.'),scope=p['scope'])
    out=R/'results/word-counterfactual-audit-v1';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 事实词改变时，已有训练轨迹是否跟着改变答案','','同一冻结任务、同一4090，全部密集/K32的0/64/128步；64/128步均两种子，零步仅一个功能基线。新1728次任务前向，复用已核验密集零步192次，0训练。','','|配置|步数|短题|无原文|32K准确率|事实两版都正确率|','|---|---:|---:|---:|---:|---:|']
    for x in summary:lines.append(f"|K{x['k']}|{x['step']}|{x['short_accuracy']:.2%}|{x['no_context_accuracy']:.2%}|{x['long_accuracy']:.2%}|{x['long_both_correct']:.2%}|")
    lines+=['','|步数|32K稀疏减密集pp|配对95%区间pp|','|---|---:|---|']
    for x in contrasts:
        e=x['long_sparse_minus_dense'];lo,hi=e['ci95_pp'];lines.append(f"|{x['step']}|{e['mean_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
    lines+=['','原文事实只差一个token；两个版本必须都答对才计配对成功。按32个背景组重采样，不把两版事实、种子、重复短输入视作独立样本。短题仅4种不同输入、无原文仅1种不同输入。本任务已用于能力筛查，结果仍是机制开发诊断；不代表独立验证、复杂阅读能力或论文已成立。','','|作业|开始UTC|结束UTC|','|---|---|---|']
    for x in control['jobs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|")
    (R/'docs/word-counterfactual-trajectory-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',summary=summary)))
if __name__=='__main__':main()
