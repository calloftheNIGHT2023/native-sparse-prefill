"""Audit every frozen0/64/128 prediction on the same4090 before interpreting trajectories."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, tarfile, numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/midpoint64-evidence-v2.tar.gz';proof=load(a.with_suffix('.json'))
    assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-midpoint64-evidence-v2';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('midpoint64-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'midpoint64-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes']
            f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/midpoint64-protocol-v2.json';assert sha(pp)==sha(R/'provenance/midpoint64-protocol-v2.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    stage=dest/'results/midpoint64-stage-v2';control=load(stage/'result.json')
    assert control['status']=='complete' and control['optimizer_updates']==0 and control['task_predictions']==1920
    assert control['protocol_sha256']==sha(pp) and len(control['jobs'])==len(p['jobs'])==15
    meta=load(dest/'data/32k-expanded-training-v0/tasks.json');seedrows=[]
    for j,run in zip(p['jobs'],control['jobs']):
        assert run['name']==j['name'] and run['returncode']==0 and run['status']=='complete'
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==0
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_midpoint64_v2.py') and sha(d/'source.py')==v['eval_source_sha256']
        assert v['checkpoint_sha256']==j['sha256'] and v['calibration_max_abs_error']<=1e-6
        label='gentle32k' if j['k']==48 else 'expanded76';mirror=R/f'results/cloud-{label}-evidence-v0'
        assert sha(mirror/j['path'])==j['sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        train=load(mirror/j['training_result']);assert v['identity']==train['identity']
        assert {n:z for n,z in v['environment'].items() if n!='gpu'}=={n:z for n,z in train['environment'].items() if n!='gpu'}
        assert v['environment']['gpu']==p['evaluation_gpu']
        cal=next(e['values'] for e in train['evaluations'] if e['step']==j['step'] and e['split']=='calibration')
        assert max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6
        selected=[x for x in meta if x['variant'] in p['variants']];pred=v['predictions']
        assert len(pred)==len(selected)==v['task_predictions']==128
        assert pred==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()]
        for x,y in zip(pred,selected):
            assert all(x[n]==y[n] for n in ['item_id','variant','gold']) and len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all()
            assert x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
        acc={t:float(np.mean([x['correct'] for x in pred if x['variant']==t])) for t in p['variants']}
        seedrows.append(dict(k=j['k'],step=j['step'],seed=j['seed'],accuracy=acc))
    summary=[]
    for k in [0,32,48]:
        for step in [0,64,128]:
            rows=[x for x in seedrows if x['k']==k and x['step']==step];assert len(rows)==(1 if step==0 else 2)
            acc={t:float(np.mean([x['accuracy'][t] for x in rows])) for t in p['variants']}
            summary.append(dict(k=k,step=step,accuracy=acc,long_minus_no_context_pp=100*(acc['long32768']-acc['no_context'])))
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,optimizer_updates=0,task_predictions=1920,summary=summary,seed_rows=seedrows,control=control,scope=p['scope'])
    out=R/'results/midpoint64-audit-v2';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 同一张4090上的固定0/64/128步评测','','旧卡迁移后的严格检查发现无原文答题差异，所以本表所有分数在同一张4090重新测量，不混合新旧卡成绩。1920次计分前向，0次训练更新。64篇旧开发文章；64/128步各两颗既有种子，0步每种注意力只计一个功能相同的零LoRA基线。','','|配置|步数|无原文|32K长题|长题减无原文|','|---|---:|---:|---:|---:|']
    for x in summary:lines.append(f"|K{x['k']}|{x['step']}|{x['accuracy']['no_context']:.2%}|{x['accuracy']['long32768']:.2%}|{x['long_minus_no_context_pp']:+.2f} pp|")
    lines+=['','该曲线用于判断后续实验方向；没有新独立测试集，不能选择最高点声称泛化，也不能从三点推断完整收敛趋势。旧训练耗时在6000Ada测量，本轮4090只评测，不用于跨卡加速比较。','','|作业|开始UTC|结束UTC|','|---|---|---|']
    for x in control['jobs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|")
    (R/'docs/midpoint64-v2-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='verified',summary=summary)))
if __name__=='__main__':main()
