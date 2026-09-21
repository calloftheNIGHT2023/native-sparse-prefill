"""Verify frozen midpoint predictions and combine every fixed endpoint."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/midpoint64-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-midpoint64-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('midpoint64-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'midpoint64-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes']
            f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/midpoint64-protocol-v0.json';assert sha(pp)==sha(R/'provenance/midpoint64-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    stage=dest/'results/midpoint64-stage-v0';control=load(stage/'result.json')
    assert control['status']=='complete' and control['optimizer_updates']==0 and control['task_predictions']==792
    assert control['protocol_sha256']==sha(pp) and len(control['jobs'])==9
    meta=load(dest/'data/32k-expanded-training-v0/tasks.json');arrays={};seedrows=[]
    for j,run in zip(p['jobs'],control['jobs']):
        assert run['name']==j['name'] and run['returncode']==0 and run['status']=='complete'
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==0
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_midpoint64.py') and sha(d/'source.py')==v['eval_source_sha256']
        assert v['checkpoint_sha256']==j['sha256'] and v['calibration_max_abs_error']<=1e-6
        label='gentle32k' if j['k']==48 else 'expanded76';mirror=R/f'results/cloud-{label}-evidence-v0'
        assert sha(mirror/j['path'])==j['sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        train=load(mirror/j['training_result']);assert v['identity']==train['identity'] and v['environment']==train['environment']
        cal=next(e['values'] for e in train['evaluations'] if e['step']==j['step']);assert max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6
        selected=[x for x in meta if x['variant'] in p['variants'] and (j['item_ids'] is None or x['item_id'] in j['item_ids'])]
        pred=v['predictions'];assert len(pred)==len(selected)==v['task_predictions']
        assert pred==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()]
        for x,y in zip(pred,selected):
            assert all(x[n]==y[n] for n in ['item_id','variant','gold']) and len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all()
            assert x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
        if j['phase']=='replay':
            assert sha(mirror/j['reference'])==j['reference_sha256']
            ref={(x['item_id'],x['variant']):x for x in load(mirror/j['reference'])['predictions']}
            for x in pred:
                old=ref[(x['item_id'],x['variant'])];assert max(abs(a-b) for a,b in zip(x['choice_logits'],old['choice_logits']))<=1e-6 and x['prediction']==old['prediction']
        else:
            by={t:np.array([int(x['correct']) for x in pred if x['variant']==t]) for t in p['variants']};assert all(len(x)==64 for x in by.values())
            arrays[(j['k'],j['seed'])]=by;seedrows.append(dict(k=j['k'],seed=j['seed'],step=64,accuracy={t:float(x.mean()) for t,x in by.items()}))
    summary=[]
    for k in [0,32,48]:
        label='gentle32k' if k==48 else 'expanded76';parent=load(R/f'results/{label}-audit-v0/result.json');assert parent['status']=='verified'
        for step in [0,64,128]:
            if step==64:
                rows=[x for x in seedrows if x['k']==k];assert len(rows)==2;acc={t:float(np.mean([x['accuracy'][t] for x in rows])) for t in p['variants']}
            else:acc=next(x['accuracy'] for x in parent['summary'] if x['k']==k and x['step']==step)
            summary.append(dict(k=k,step=step,accuracy={t:acc[t] for t in p['variants']}))
    out=R/'results/midpoint64-audit-v0';out.mkdir(exist_ok=True)
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,optimizer_updates=0,task_predictions=792,midpoint_predictions=768,replay_predictions=24,summary=summary,seed_rows=seedrows,control=control,scope=p['scope'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 32K固定中点：不重训，观察0/64/128步任务变化','','全部64篇旧开发文章，密集/K32/K48各两颗既有种子。三种注意力路径先重放已有128步模型的24个选项logit，全部校准值重放后才接纳中点数据。0新增更新，792计分前向，其中24个用于实现核验。','','|配置|步数|无原文|32K长题|','|---|---:|---:|---:|']
    for x in summary:lines.append(f"|K{x['k']}|{x['step']}|{x['accuracy']['no_context']:.2%}|{x['accuracy']['long32768']:.2%}|")
    lines+=['','固定展示所有结果，不选择最优断点作泛化声明。0/64/128三个点只能描述粗粒度变化，不能证明完整收敛趋势、稀疏理论成立或定位路由因果根因。没有新增训练，也没有从这组开发数据中确定非劣界限。','','|作业|开始UTC|结束UTC|','|---|---|---|']
    for x in control['jobs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|")
    (R/'docs/midpoint64-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',summary=summary)))
if __name__=='__main__':main()
