"""Verify raw option-rotation evidence; use articles, not prompts, as bootstrap units."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/option-rotation-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-option-rotation-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('option-rotation-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'option-rotation-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes']
            f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/option-rotation-protocol-v0.json';assert sha(pp)==sha(R/'provenance/option-rotation-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    assert sha(R/'results/paired-position-audit-v0/result.json')==p['parent_audit_sha256']
    stage=dest/'results/option-rotation-stage-v0';control=load(stage/'result.json')
    assert control['status']=='complete' and control['optimizer_updates']==0 and control['task_predictions']==2048
    assert control['protocol_sha256']==sha(pp) and len(control['jobs'])==len(p['jobs'])==4
    meta=load(dest/'data/option-rotation-v0/tasks.json');ids=sorted({x['item_id'] for x in meta});positions=[0,1,2,3]
    assert len(ids)==64 and len(meta)==512
    data={};semantic={};replay_errors=[]
    for j,run in zip(p['jobs'],control['jobs']):
        assert run['name']==j['name'] and run['returncode']==0 and run['status']=='complete'
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==0
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_option_rotation_v0.py')==sha(d/'source.py')
        assert v['checkpoint_sha256']==j['sha256'] and v['calibration_max_abs_error']<=1e-6 and v['replay_logit_max_abs_error']<=1e-6
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        train=load(mirror/j['training_result']);assert v['identity']==train['identity']
        assert {n:z for n,z in v['environment'].items() if n!='gpu'}=={n:z for n,z in train['environment'].items() if n!='gpu'} and v['environment']['gpu']==p['evaluation_gpu']
        cal=next(e['values'] for e in train['evaluations'] if e['step']==128 and e['split']=='calibration');assert max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6
        assert sha(dest/j['reference'])==j['reference_sha256']==sha(R/'results/cloud-midpoint64-evidence-v2'/j['reference'])
        previous=load(dest/j['reference']);assert previous['environment']==v['environment']
        ref={x['item_id']:x for x in previous['predictions'] if x['variant']=='long32768'}
        pred=v['predictions'];assert len(pred)==v['task_predictions']==512
        assert pred==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()]
        arr=np.full((64,4,2),np.nan);answers=np.full((64,4,2),-1,dtype=int);replays=0
        for x,y in zip(pred,meta):
            assert all(x[n]==y[n] for n in ['item_id','variant','gold','rotation','original_gold','original_replay'])
            assert len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all() and x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
            i=ids.index(x['item_id']);z=x['rotation'];h=0 if x['variant']=='short' else 1
            assert np.isnan(arr[i,z,h]);arr[i,z,h]=int(x['correct']);assert x['semantic_prediction']==(x['prediction']+x['rotation'])%4;answers[i,z,h]=x['semantic_prediction']
            assert x['correct']==(x['semantic_prediction']==x['original_gold'])
            if x['original_replay']:
                old=ref[x['item_id']];error=max(abs(a-b) for a,b in zip(x['choice_logits'],old['choice_logits']));assert error<=1e-6 and x['prediction']==old['prediction'];replay_errors.append(error);replays+=1
        assert replays==64 and np.isfinite(arr).all();data[(j['k'],j['seed'])]=arr;semantic[(j['k'],j['seed'])]=answers
    assert len(data)==4
    rng=np.random.default_rng(2026091691);draws=rng.integers(0,64,size=(10000,64))
    def estimate(article_scores):
        assert article_scores.shape==(64,)
        return dict(mean_pp=float(article_scores.mean()*100),ci95_pp=(np.quantile(article_scores[draws].mean(axis=1),[.025,.975])*100).tolist())
    means={k:np.mean([data[(k,s)] for s in [2026091660,2026091661]],axis=0) for k in [0,32]}
    summary=[]
    for variant,h in [('short',0),('long32768',1)]:
        parts={k:x[:,:,h] for k,x in means.items()}
        methods={}
        for k,x in parts.items():
            consistent=np.mean([np.all(semantic[(k,seed)][:,:,h]==semantic[(k,seed)][:,0:1,h],axis=1) for seed in [2026091660,2026091661]],axis=0)
            methods[str(k)]=dict(original_order_accuracy=float(x[:,0].mean()),uniform_four_order_accuracy=float(x.mean()),change=estimate(x.mean(axis=1)-x[:,0]),semantic_consistency=float(consistent.mean()),accuracy_by_rotation=x.mean(axis=0).tolist())
        original=parts[32][:,0]-parts[0][:,0];balanced=parts[32].mean(axis=1)-parts[0].mean(axis=1)
        summary.append(dict(variant=variant,methods=methods,original_sparse_minus_dense=estimate(original),balanced_sparse_minus_dense=estimate(balanced),gap_change=estimate(balanced-original)))
    out=R/'results/option-rotation-audit-v0';out.mkdir(exist_ok=True)
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,optimizer_updates=0,task_predictions=2048,exact_replay_predictions=256,replay_max_abs_error=max(replay_errors),summary=summary,bootstrap=dict(unit='article',n=64,draws=10000,seed=2026091691,scope='Exploratory intervals jointly preserving bothseeds andallrotations; no independent-confirmation claim.'),control=control,scope=p['scope'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 四种选项循环排列的完整诊断','','64篇旧开发文章、四种循环排列、短题与32K题；密集/K32各两颗128步种子。2048计分前向、0训练。256条原顺序长题按原1e-6门槛重放，所有校准/模型/数据/源码/逐题分数核验。','','|输入|配置|原顺序|四排列均值|语义答案四排列一致率|','|---|---|---:|---:|---:|']
    for row in summary:
        for k,v in row['methods'].items():lines.append(f"|{row['variant']}|K{k}|{v['original_order_accuracy']:.2%}|{v['uniform_four_order_accuracy']:.2%}|{v['semantic_consistency']:.2%}|")
    lines+=['','|输入|原顺序差pp|均衡后差pp|均衡后差95%区间pp|差距变化pp|','|---|---:|---:|---|---:|']
    for row in summary:
        e=row['balanced_sparse_minus_dense'];lo,hi=e['ci95_pp'];lines.append(f"|{row['variant']}|{row['original_sparse_minus_dense']['mean_pp']:+.2f}|{e['mean_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|{row['gap_change']['mean_pp']:+.2f}|")
    lines+=['','差值均为K32减密集。四排列等权平均是稳健评测统计，不是只跑一次能获得的集成准确率。未挑最好排列、未拟合校准、未筛题；每篇的原文/背景/问题和总token数完全不变。选项顺序和字母偏置已有研究，本轮不声明原创；这些仍是旧开发集上的描述性结果，不是独立非劣性或长文理解证据。','','|作业|开始UTC|结束UTC|','|---|---|---|']
    for x in control['jobs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|")
    (R/'docs/option-rotation-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',summary=summary)))
if __name__=='__main__':main()
