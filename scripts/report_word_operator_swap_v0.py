"""Verify full fixed trajectories on the frozen assigned-word task."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/word-operator-swap-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-word-operator-swap-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('word-operator-swap-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'word-operator-swap-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/word-operator-swap-protocol-v0.json';assert sha(pp)==sha(R/'provenance/word-operator-swap-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    gate=R/'results/word-counterfactual-audit-v1/result.json';assert sha(gate)==p['parent_trajectory_audit_sha256'] and load(gate)['status']=='verified'
    stage=dest/'results/word-operator-swap-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and control['optimizer_updates']==0 and control['task_predictions']==768 and control['protocol_sha256']==sha(pp)
    assert len(control['jobs'])==len(p['jobs'])==4
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
        j=v['job'];arrays[(j['k'],v.get('evaluation_k',j['k']),j['seed'])]=by
        seedrows.append(dict(k=j['k'],evaluation_k=v.get('evaluation_k',j['k']),step=j['step'],seed=j['seed'],short_accuracy=float(by['short'].mean()),no_context_accuracy=float(by['no_context'].mean()),long_accuracy=float(by['long32768'].mean()),long_both_correct=float(np.all(by['long32768']==1,axis=1).mean())))
    base=None
    for ref in p['native_references']:
        f=dest/ref['path'];assert sha(f)==ref['sha256']==sha(R/'results/cloud-word-counterfactual-evidence-v1'/ref['path'])
        v=load(f);assert v['job']['k']==ref['k'] and v['job']['seed']==ref['seed'] and v['job']['step']==128;add(v)
        if base is None:base=v
        else:assert v['environment']==base['environment']
    for j,run in zip(p['jobs'],control['jobs']):
        assert run['name']==j['name'] and run['returncode']==0 and run['status']=='complete'
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==0 and v['step']==j['step'] and v['evaluation_k']==j['evaluation_k']
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_word_operator_swap_v0.py')==sha(d/'source.py')
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        train=load(mirror/j['training_result']);assert v['identity']==train['identity'] and v['environment']==base['environment']
        cal=next(e['values'] for e in train['evaluations'] if e['step']==j['step'] and e['split']=='calibration');assert max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        assert v['predictions']==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()];add(v)
    assert len(arrays)==8
    rng=np.random.default_rng(2026091695);draws=rng.integers(0,32,size=(10000,32))
    def estimate(a):return dict(mean_pp=float(a.mean()*100),ci95_pp=(np.quantile(a[draws].mean(axis=1),[.025,.975])*100).tolist())
    summary=[];points={}
    for training_k in [0,32]:
        for evaluation_k in [0,32]:
            selected=[v for (kk,ek,_),v in arrays.items() if kk==training_k and ek==evaluation_k];assert len(selected)==2
            mean=np.mean([v['long32768'].mean(axis=1) for v in selected],axis=0);both=np.mean([np.all(v['long32768']==1,axis=1) for v in selected],axis=0);points[(training_k,evaluation_k)]=mean
            summary.append(dict(training_k=training_k,evaluation_k=evaluation_k,short_accuracy=float(np.mean([v['short'].mean() for v in selected])),no_context_accuracy=float(np.mean([v['no_context'].mean() for v in selected])),long_accuracy=float(mean.mean()),long_both_correct=float(both.mean())))
    contrasts={name:estimate(points[a]-points[b]) for name,a,b in [('sparse_weights_switch_to_dense',(32,0),(32,32)),('dense_weights_switch_to_sparse',(0,32),(0,0)),('sparse_minus_dense_weights_under_dense',(32,0),(0,0)),('sparse_minus_dense_weights_under_sparse',(32,32),(0,32))]}
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,optimizer_updates=0,task_predictions=768,reused_native_predictions=768,summary=summary,seed_rows=seedrows,contrasts=contrasts,control=control,bootstrap=dict(unit='background_family',n=32,draws=10000,seed=2026091695,scope='Conditional exploratory intervals preserving factualpairs andbothseeds. No independent confirmation.'),scope=p['scope'])
    out=R/'results/word-operator-swap-audit-v0';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 固定权重交换注意力：直接事实词任务','','固定128步、两个种子、32组背景。所有原生注意力校准NLL通过1e-6后再交换评测算子。新768次计分前向、复用768次原生预测，0训练。','','|训练注意力|评测注意力|短题|32K准确率|事实两版都正确率|','|---|---|---:|---:|---:|']
    for x in summary:lines.append(f"|K{x['training_k']}|K{x['evaluation_k']}|{x['short_accuracy']:.2%}|{x['long_accuracy']:.2%}|{x['long_both_correct']:.2%}|")
    lines+=['','K0为密集。','','|固定对照|变化pp|背景组配对95%区间pp|','|---|---:|---|']
    for name,e in contrasts.items():
        lo,hi=e['ci95_pp'];lines.append(f"|{name}|{e['mean_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
    lines+=['','这是固定模型参数的完整2x2算子干预，未做训练。32背景、两个事实版本和两个种子按背景组共同重采样。短输入只含4种不同模板，不视为64个独立题。该任务已用于能力筛查；结果只能定位这批任务中的机制，不能当独立验证、新方法或加速证据。此前RACE回切未恢复的结果仍保留。','','|作业|开始UTC|结束UTC|','|---|---|---|']
    for x in control['jobs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|")
    (R/'docs/word-operator-swap-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',summary=summary,contrasts=contrasts)))
if __name__=='__main__':main()
