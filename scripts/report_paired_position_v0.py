"""Verify raw paired-position evidence; use articles, not prompts, as bootstrap units."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/paired-position-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-paired-position-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('paired-position-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'paired-position-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes']
            f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/paired-position-protocol-v0.json';assert sha(pp)==sha(R/'provenance/paired-position-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    assert sha(R/'results/midpoint64-audit-v2/result.json')==p['parent_audit_sha256']
    stage=dest/'results/paired-position-stage-v0';control=load(stage/'result.json')
    assert control['status']=='complete' and control['optimizer_updates']==0 and control['task_predictions']==2048
    assert control['protocol_sha256']==sha(pp) and len(control['jobs'])==len(p['jobs'])==4
    meta=load(dest/'data/paired-position-v0/tasks.json');ids=sorted({x['item_id'] for x in meta});positions=[.1,.35,.65,.9]
    assert len(ids)==64 and len(meta)==512
    data={};replay_errors=[]
    for j,run in zip(p['jobs'],control['jobs']):
        assert run['name']==j['name'] and run['returncode']==0 and run['status']=='complete'
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==0
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_paired_position_v0.py')==sha(d/'source.py')
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
        arr=np.full((64,4,2),np.nan);replays=0
        for x,y in zip(pred,meta):
            assert all(x[n]==y[n] for n in ['item_id','variant','gold','condition','evidence_fraction','original_replay'])
            assert len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all() and x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
            i=ids.index(x['item_id']);z=positions.index(x['evidence_fraction']);h=0 if x['condition']=='present' else 1
            assert np.isnan(arr[i,z,h]);arr[i,z,h]=int(x['correct'])
            if x['original_replay']:
                old=ref[x['item_id']];error=max(abs(a-b) for a,b in zip(x['choice_logits'],old['choice_logits']));assert error<=1e-6 and x['prediction']==old['prediction'];replay_errors.append(error);replays+=1
        assert replays==64 and np.isfinite(arr).all();data[(j['k'],j['seed'])]=arr
    assert len(data)==4
    rng=np.random.default_rng(2026091691);draws=rng.integers(0,64,size=(10000,64))
    def estimate(article_scores):
        assert article_scores.shape==(64,)
        return dict(mean_pp=float(article_scores.mean()*100),ci95_pp=(np.quantile(article_scores[draws].mean(axis=1),[.025,.975])*100).tolist())
    means={k:np.mean([data[(k,s)] for s in [2026091660,2026091661]],axis=0) for k in [0,32]}
    summary=[]
    for name,z in [('all_positions',None)]+[(str(v),i) for i,v in enumerate(positions)]:
        parts={k:(x.mean(axis=1) if z is None else x[:,z,:]) for k,x in means.items()}
        summary.append(dict(position=name,methods={str(k):dict(present_accuracy=float(x[:,0].mean()),ablated_accuracy=float(x[:,1].mean()),evidence_effect=estimate(x[:,0]-x[:,1])) for k,x in parts.items()},sparse_minus_dense_present=estimate(parts[32][:,0]-parts[0][:,0]),sparse_minus_dense_evidence_effect=estimate((parts[32][:,0]-parts[32][:,1])-(parts[0][:,0]-parts[0][:,1]))))
    out=R/'results/paired-position-audit-v0';out.mkdir(exist_ok=True)
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,optimizer_updates=0,task_predictions=2048,exact_replay_predictions=256,replay_max_abs_error=max(replay_errors),summary=summary,bootstrap=dict(unit='article',n=64,draws=10000,seed=2026091691,scope='Exploratory conditional intervals preserving bothseeds andallpositions; no population-of-training-seeds or independent-confirmation claim.'),control=control,scope=p['scope'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 同文、同长度的位置与原文替换诊断','','64篇旧开发文章，每篇四个位置、两种原文条件；密集/K32各两颗128步种子。2048计分前向、0训练更新。256条原位置输入按原1e-6阈值重放；数据/模型/源码/逐题分数均核验。','','|位置|配置|原文存在|等长替换|原文作用pp|配对95%区间pp|','|---|---|---:|---:|---:|---|']
    for row in summary:
        for k,v in row['methods'].items():
            e=v['evidence_effect'];lo,hi=e['ci95_pp'];lines.append(f"|{row['position']}|K{k}|{v['present_accuracy']:.2%}|{v['ablated_accuracy']:.2%}|{e['mean_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
    lines+=['','|位置|K32减密集：原文存在准确率差pp|K32减密集：原文作用差pp|原文作用差95%区间pp|','|---|---:|---:|---|']
    for row in summary:
        e=row['sparse_minus_dense_evidence_effect'];lo,hi=e['ci95_pp'];lines.append(f"|{row['position']}|{row['sparse_minus_dense_present']['mean_pp']:+.2f}|{e['mean_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
    lines+=['','原文替换仍保留TARGET标记，内容换成等token数的固定无关背景。它是受控输入消融，不是语义答案反事实；不能仅据差异推断路由因果根因。所有位置共同按文章重采样，不把位置和种子当独立文章，也不选最有利位置。这是旧开发集探索性诊断，不是独立验证或非劣性证明。','','|作业|开始UTC|结束UTC|','|---|---|---|']
    for x in control['jobs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|")
    (R/'docs/paired-position-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',summary=summary)))
if __name__=='__main__':main()
