"""Audit paired whole-model prefill timings; never infer training savings."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/word-suffix-cost-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-word-suffix-cost-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('word-suffix-cost-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'word-suffix-cost-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/word-suffix-cost-protocol-v0.json';assert sha(pp)==sha(R/'provenance/word-suffix-cost-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    assert sha(R/'results/word-suffix-audit-v0/result.json')==p['parent_audit_sha256']
    stage=dest/'results/word-suffix-cost-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and control['optimizer_updates']==0 and control['task_predictions']==300 and control['protocol_sha256']==sha(pp)
    assert len(control['jobs'])==len(p['jobs'])==4 and control['seconds']<=p['maximum_seconds']
    meta={x['item_id']:x for x in load(dest/'data/word-counterfactual-v0/tasks.json') if x['variant']=='long32768'};refs={};rounds=[];modelrows=[]
    for ref in p['references']:
        f=dest/ref['path'];assert sha(f)==ref['sha256'];refs[ref['training_k'],ref['seed'],ref['condition']]=load(f)
    for j,run in zip(p['jobs'],control['jobs']):
        assert run['name']==j['name'] and run['returncode']==0 and run['status']=='complete'
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==0 and v['step']==128 and v['task_predictions']==75
        assert v['eval_source_sha256']==sha(dest/'scripts/bench_word_suffix_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu']
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        train=load(mirror/j['training_result']);assert v['identity']==train['identity']
        cal=next(e['values'] for e in train['evaluations'] if e['step']==128 and e['split']=='calibration');assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        preds=v['predictions'];assert len(preds)==75 and preds==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()]
        for x,planned in zip(preds,p['schedule']):
            assert all(x[n]==value for n,value in planned.items()) and all(x[n]==value for n,value in meta[x['item_id']].items())
            rv=refs[j['k'],j['seed'],x['mode']];assert rv['environment']==v['environment'];old=next(y for y in rv['predictions'] if y['variant']=='long32768' and y['item_id']==x['item_id'])
            assert np.isfinite(x['choice_logits']).all() and len(x['choice_logits'])==4 and max(abs(a-b) for a,b in zip(x['choice_logits'],old['choice_logits']))<=1e-6
            assert x['prediction']==old['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold']) and x['use_cache'] is True and x['replay_max_abs_error']<=1e-6
            assert x['wall_seconds']>0 and x['cuda_ms']>0 and x['peak_allocated_bytes']>0
        for rep in range(3):
            by={m:[x for x in preds if x['phase']=='timed' and x['round']==rep and x['mode']==m] for m in ['DD','SS','SD']};assert all(len(xs)==8 for xs in by.values())
            wall={m:sum(x['wall_seconds'] for x in xs) for m,xs in by.items()};gpu={m:sum(x['cuda_ms'] for x in xs) for m,xs in by.items()}
            rounds.append(dict(k=j['k'],seed=j['seed'],round=rep,wall_seconds_sum=wall,cuda_ms_sum=gpu,SD_saving_percent=(1-wall['SD']/wall['DD'])*100,SS_saving_percent=(1-wall['SS']/wall['DD'])*100,SD_vs_SS_overhead_percent=(wall['SD']/wall['SS']-1)*100))
        modelrows.append(dict(k=j['k'],seed=j['seed'],mode_stats={m:dict(median_wall_seconds=float(np.median([x['wall_seconds'] for x in preds if x['phase']=='timed' and x['mode']==m])),peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in preds if x['mode']==m)) for m in ['DD','SS','SD']},gpu_before=v['gpu_before'],gpu_after=v['gpu_after']))
    savings=[x['SD_saving_percent'] for x in rounds];summary=dict(median_round_saving_percent=float(np.median(savings)),min_round_saving_percent=min(savings),max_round_saving_percent=max(savings),all_rounds_positive=all(x>0 for x in savings));summary['cost_screen_passed']=summary['median_round_saving_percent']>=p['cost_screen']['minimum_median_round_saving_percent'] and summary['all_rounds_positive']
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,optimizer_updates=0,task_predictions=300,warmup_predictions=12,timed_predictions=288,summary=summary,model_rows=modelrows,rounds=rounds,control=control,scope=p['scope'])
    out=R/'results/word-suffix-cost-audit-v0';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 密集问题后缀：整模型prefill耗时核验','','同一4090、batch1、32K、生成KV缓存并计算最终位置logits。输入已在GPU；不包含分词、传输、排队、网络和模型加载，不等同服务TTFT或训练成本。三个模式交错运行，包含路由和重复计算后丢弃的稀疏后缀。全部300次输出按原1e-6阈值重放；12次预热不计入时间统计。','','|训练K|种子|轮次|密集总秒|稀疏总秒|稀疏+密集后缀总秒|后缀方案比密集省时|','|---|---|---|---:|---:|---:|---:|']
    for x in rounds:
        w=x['wall_seconds_sum'];lines.append(f"|{x['k']}|{x['seed']}|{x['round']}|{w['DD']:.4f}|{w['SS']:.4f}|{w['SD']:.4f}|{x['SD_saving_percent']:.2f}%|")
    lines+=['',f"12个模型-轮次省时中位数 {summary['median_round_saving_percent']:.2f}%，范围 [{summary['min_round_saving_percent']:.2f}%, {summary['max_round_saving_percent']:.2f}%]。资源分配成本筛查通过：{summary['cost_screen_passed']}。",'','这是开发阶段成本筛查，不是质量等价或论文结论。保留完整质量表中的差距；不能混用旧RTX6000Ada训练时间推导本机训练加速，也不能把反复计时当独立模型样本。','','|作业|开始UTC|结束UTC|','|---|---|---|']
    for x in control['jobs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|")
    (R/'docs/word-suffix-cost-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',summary=summary)))
if __name__=='__main__':main()
