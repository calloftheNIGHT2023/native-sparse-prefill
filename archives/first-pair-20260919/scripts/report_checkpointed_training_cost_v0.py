"""Verify matched-checkpointing training-body costs, without inferring quality."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/checkpointed-training-cost-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-checkpointed-training-cost-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('checkpointed-training-cost-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'checkpointed-training-cost-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/checkpointed-training-cost-protocol-v0.json';assert sha(pp)==sha(R/'provenance/checkpointed-training-cost-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    assert sha(R/'results/word-suffix-profile-audit-v0/result.json')==p['parent_audit_sha256']
    stage=dest/'results/checkpointed-training-cost-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and control['task_predictions']==0 and control['gradient_passes']==18 and control['optimizer_updates']==0 and control['protocol_sha256']==sha(pp)
    assert len(control['jobs'])==len(p['jobs'])==1;job=p['jobs'][0];d=stage/job['name'];v=load(d/'result.json')
    assert v['status']=='complete' and v['job']==job and v['gradient_passes']==18 and v['optimizer_updates']==v['task_predictions']==0 and v['step']==128
    assert v['eval_source_sha256']==sha(dest/'scripts/benchmark_checkpointed_training_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu']
    mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/job['path'])==job['sha256']==v['checkpoint_sha256'] and sha(mirror/job['training_result'])==job['training_result_sha256']
    train=load(mirror/job['training_result']);assert train['identity']==v['identity'];cal=next(e['values'] for e in train['evaluations'] if e['step']==128 and e['split']=='calibration');assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
    rows=v['rows'];assert len(rows)==18 and rows==[json.loads(x) for x in (d/'gradient-passes.jsonl').read_text().splitlines()]
    assert v['weights_unchanged'] and v['initial_parameter_digest']==v['final_parameter_digest'];assert len(v['gates'])==8 and all(g['passed'] for g in v['gates'])
    for g in v['gates']:
        if g.get('name')=='repeat_gradient':assert g['loss_gap']<.001
        else:assert g['loss_gap']<.001 and g['gradient_relative_l2']<.02
    training=np.load(dest/'data/32k-expanded-training-v0/train-calibration.npz')['train'];plan=[]
    for k in [0,32]:
        for cp in [False,True]:plan.append(dict(k=k,checkpointing=cp,phase='math_gate',window=0,tokens=2048,round=-1,repeat=-1))
    for k in [0,32]:plan.append(dict(k=k,checkpointing=True,phase='warmup',window=0,tokens=32768,round=-1,repeat=-1))
    for rep,order in enumerate(p['orders']):
        for k in order:
            for repeat in [0,1]:plan.append(dict(k=k,checkpointing=True,phase='timed',window=p['windows'][rep],tokens=32768,round=rep,repeat=repeat))
    for x,expected in zip(rows,plan):
        assert all(x[k]==value for k,value in expected.items());w=training[x['window'],:x['tokens']+1];assert hashlib.sha256(w.tobytes()).hexdigest()==x['input_sha256']
        assert all(np.isfinite(x[k]) and x[k]>0 for k in ['seconds','forward_cuda_ms','head_and_backbone_backward_cuda_ms','peak_allocated_bytes']) and np.isfinite(x['loss']) and np.isfinite(x['gradient_norm'])
    comparisons=[]
    for rep in range(3):
        by={k:[x for x in rows if x['phase']=='timed' and x['round']==rep and x['k']==k] for k in [0,32]};assert all(len(z)==2 for z in by.values())
        dense=float(np.median([x['seconds'] for x in by[0]]));sparse=float(np.median([x['seconds'] for x in by[32]]));comparisons.append(dict(round=rep,dense_seconds=dense,sparse_seconds=sparse,saving_percent=(1-sparse/dense)*100,phases={k:{n:float(np.median([x[n] for x in z])) for n in ['forward_cuda_ms','head_and_backbone_backward_cuda_ms','clip_cuda_ms','peak_allocated_bytes']} for k,z in by.items()}))
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,optimizer_updates=0,task_predictions=0,gradient_passes=18,comparisons=comparisons,median_saving_percent=float(np.median([x['saving_percent'] for x in comparisons])),weights_unchanged=True,control=control,scope=p['scope']);out=R/'results/checkpointed-training-cost-audit-v0';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 4090匹配梯度检查点的训练主体成本','','同一固定权重，两组都启用非重入梯度检查点，32K输入，FP32参数/BF16 autocast、分块输出头反传及裁剪；未执行优化器步骤。2048-token开启/关闭检查点的损失和梯度检查通过，32K重复梯度检查通过；参数哈希未变。18次梯度计算、0参数更新、0任务评测。','','|窗口/轮次|密集秒|K32秒|K32省时|密集峰值GiB|K32峰值GiB|','|---|---:|---:|---:|---:|---:|']
    for x in comparisons:lines.append(f"|{x['round']}|{x['dense_seconds']:.4f}|{x['sparse_seconds']:.4f}|{x['saving_percent']:.2f}%|{x['phases'][0]['peak_allocated_bytes']/2**30:.2f}|{x['phases'][32]['peak_allocated_bytes']/2**30:.2f}|")
    lines+=['',f"三轮省时中位数：{result['median_saving_percent']:.2f}%。",'','时间包含骨干前向、分块头反传、骨干反向、检查点重算与裁剪；不包含优化器、数据传输/加载和CPU梯度校验。不能据此声称完整训练任务省钱或质量等价；也不是旧RTX6000Ada未启用检查点的原设置。质量仍需沿用已有审计中的差距。']
    (R/'docs/checkpointed-training-cost-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',comparisons=comparisons,median_saving_percent=result['median_saving_percent'])))
if __name__=='__main__':main()
