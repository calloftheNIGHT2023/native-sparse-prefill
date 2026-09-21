"""Audit complete optimizer-step timing; explicitly a diagnostic, not convergence proof."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/full-step-cost-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-full-step-cost-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('full-step-cost-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'full-step-cost-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/full-step-cost-protocol-v0.json';assert sha(pp)==sha(R/'provenance/full-step-cost-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    assert sha(R/'results/checkpointed-training-cost-audit-v0/result.json')==p['numerical_preflight_audit_sha256']
    stage=dest/'results/full-step-cost-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and control['task_predictions']==control['gradient_passes']==0 and control['optimizer_updates']==64 and control['protocol_sha256']==sha(pp)
    train=np.load(dest/'data/32k-expanded-training-v0/train-calibration.npz')['train'];order=np.random.default_rng(2026091662).permutation(76).tolist();by={};records=[]
    for j in p['jobs']:
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==16 and v['task_predictions']==0 and v['step']==144
        assert v['eval_source_sha256']==sha(dest/'scripts/benchmark_full_step_cost_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu']
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        old=load(mirror/j['training_result']);assert old['identity']==v['identity']['parent'] and v['identity']['training_k']==j['training_k'] and v['identity']['protocol_sha256']==sha(pp)
        cal=next(e['values'] for e in old['evaluations'] if e['step']==128 and e['split']=='calibration');assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        assert sha(d/'checkpoint-144.pt')==v['final_checkpoint_sha256']
        rows=v['rows'];assert len(rows)==16 and rows==[json.loads(x) for x in (d/'training-steps.jsonl').read_text().splitlines()]
        for i,row in enumerate(rows,128):
            wi=order[i%76];assert row['step']==i+1 and row['window_index']==wi and row['input_sha256']==hashlib.sha256(train[wi].tobytes()).hexdigest() and row['training_k']==j['training_k']
            assert abs(row['lr']-.0001)<1e-12 and all(np.isfinite(row[n]) for n in ['loss','gradient_norm','seconds']) and row['seconds']>0
        assert abs(v['complete_step_seconds']-sum(x['seconds'] for x in rows))<1e-8
        by[(j['seed'],j['training_k'])]=v
        records.append(dict(seed=j['seed'],training_k=j['training_k'],all16_step_seconds=v['complete_step_seconds'],loop_seconds=v['loop_seconds'],steady14_median_seconds=float(np.median([x['seconds'] for x in rows[2:]])),peak_gib=v['peak_allocated_bytes']/2**30,final_common_dense_calibration_mean=float(np.mean(v['final_common_dense_calibration_values']))))
    comparisons=[]
    for seed in [2026091660,2026091661]:
        d=by[(seed,0)];s=by[(seed,32)];comparisons.append(dict(seed=seed,all16_step_saving_percent=100*(1-s['complete_step_seconds']/d['complete_step_seconds']),loop_saving_percent=100*(1-s['loop_seconds']/d['loop_seconds']),whole_job_saving_percent=100*(1-s['seconds']/d['seconds'])))
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,optimizer_updates=64,scientific_updates=0,diagnostic_updates=64,task_predictions=0,records=records,comparisons=comparisons,scope=p['scope']);out=R/'results/full-step-cost-audit-v0';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 匹配设置下的完整训练步成本','','两种子各从同一个密集128步检查点及优化器状态分叉，分别做16次密集或K32更新；两组都用非重入梯度检查点。64次均计为诊断更新，不计入正式质量训练。不是从第0步全程稀疏的质量证明。','','|种子|训练K|16步合计秒|循环秒（含逐步日志）|后14步中位秒|峰值GiB|','|---|---:|---:|---:|---:|---:|']
    for x in records:lines.append(f"|{x['seed']}|{x['training_k']}|{x['all16_step_seconds']:.3f}|{x['loop_seconds']:.3f}|{x['steady14_median_seconds']:.4f}|{x['peak_gib']:.2f}|")
    for x in comparisons:lines+=['',f"种子{x['seed']}：完整16步省时{x['all16_step_saving_percent']:.2f}%，含日志循环省时{x['loop_saving_percent']:.2f}%，进程内总作业省时{x['whole_job_saving_percent']:.2f}%。"]
    lines+=['','主计时包含清梯度、CPU至GPU输入创建/传输、完整前反向及检查点重算、梯度裁剪、AdamW和调度器更新及同步；循环计时还包含逐步日志。模型/数据加载、校准检查和检查点写盘包含在进程内总作业耗时，但不在逐步计时内。整个研究费用还包括准备、空闲和存储，不能将本轮节省直接外推为达到同质量所需总费用。只有两个串行重复；保留冷启动两步，不以丢弃慢步后的数值充当主结果。']
    (R/'docs/full-step-cost-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',records=records,comparisons=comparisons)))
if __name__=='__main__':main()
