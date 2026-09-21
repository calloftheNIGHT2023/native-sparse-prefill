"""Verify archived common-dense NLL diagnosis and compute paired descriptive contrasts."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/common-dense-quality-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-common-dense-quality-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('common-dense-quality-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'common-dense-quality-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/common-dense-quality-protocol-v0.json';assert sha(pp)==sha(R/'provenance/common-dense-quality-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    stage=dest/'results/common-dense-quality-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and control['task_predictions']==control['optimizer_updates']==control['gradient_passes']==0 and control['protocol_sha256']==sha(pp)
    assert len(control['jobs'])==len(p['jobs'])==9
    windows=np.load(dest/'data/32k-expanded-training-v0/report.npz')['report'];records=[];by={}
    for j in p['jobs']:
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['evaluation_k']==0 and v['optimizer_updates']==v['task_predictions']==0 and v['nll_forwards']==31
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_common_dense_quality_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu'] and v['weights_unchanged']
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        train=load(mirror/j['training_result']);assert train['identity']==v['identity'];cal=next(e['values'] for e in train['evaluations'] if e['step']==j['step'] and e['split']=='calibration')
        assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        rows=v['windows'];assert len(rows)==9 and rows==[json.loads(x) for x in (d/'window-nll.jsonl').read_text().splitlines()]
        for i,(row,w) in enumerate(zip(rows,windows)):
            assert row['window']==i and row['input_sha256']==hashlib.sha256(w.tobytes()).hexdigest()
            assert all(np.isfinite(row[n]) for n in ['full_nll','full_context_tail_nll','short_context_tail_nll','context_benefit_nats'])
            assert abs(row['context_benefit_nats']-(row['short_context_tail_nll']-row['full_context_tail_nll']))<1e-12
        by[(j['k'],j['seed'],j['step'])]=v
        record={k:j[k] for k in ['k','seed','step']};record.update({n:float(np.mean([x[n] for x in rows])) for n in ['full_nll','context_benefit_nats']});record['perplexity']=float(np.exp(record['full_nll']));records.append(record)
    rng=np.random.default_rng(2026091671);idx=rng.integers(0,9,(20000,9));contrasts=[]
    for step in [64,128]:
        for metric in ['full_nll','context_benefit_nats']:
            diffs=np.array([[s[metric]-d[metric] for s,d in zip(by[(32,seed,step)]['windows'],by[(0,seed,step)]['windows'])] for seed in [2026091660,2026091661]])
            mean=diffs.mean(axis=0);ci=np.quantile(mean[idx].mean(axis=1),[.025,.975]).tolist()
            x=dict(step=step,metric=metric,sparse_minus_dense=float(mean.mean()),by_seed=diffs.mean(axis=1).tolist(),conditional_window_bootstrap_95=ci)
            if metric=='full_nll':x.update(perplexity_ratio=float(np.exp(mean.mean())),perplexity_ratio_ci=np.exp(ci).tolist())
            contrasts.append(x)
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,optimizer_updates=0,task_predictions=0,nll_forwards=279,records=records,contrasts=contrasts,scope=p['scope'])
    out=R/'results/common-dense-quality-audit-v0';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 统一密集注意力评测：训练权重质量诊断','','所有模型使用同一密集评测算子。9个旧WikiText报告窗口已多次使用，属于开发诊断，不是独立确认。两种子、0/64/128检查点；每个检查点先验证原生校准NLL误差不超过1e-6。没有参数更新。','','|训练K|种子|步数|完整32K NLL|困惑度|长上下文收益（nats）|','|---|---:|---:|---:|---:|---:|']
    for x in records:lines.append(f"|{x['k']}|{x['seed']}|{x['step']}|{x['full_nll']:.6f}|{x['perplexity']:.4f}|{x['context_benefit_nats']:.6f}|")
    for x in contrasts:lines+=['',f"{x['step']}步 {x['metric']}：稀疏减密集 {x['sparse_minus_dense']:.6f}；按9个窗口重采样的条件95%区间 {x['conditional_window_bootstrap_95']}。"+(f" 困惑度比值 {x['perplexity_ratio']:.6f}。" if 'perplexity_ratio' in x else '')]
    lines+=['','NLL越低越好；长上下文收益为同一末尾4096个目标在短上下文和完整32K上下文下的NLL差，正值表示长上下文有帮助。区间只描述这9个窗口；窗口不能视为独立文档，两个种子也不足以推断广泛训练稳定性。不同注意力下的旧NLL不直接用来计算当前差值。普通文本预测相近不能替代长文本事实读取质量。训练主体成本7.85%的节省尚不代表完整训练任务总成本节省。']
    (R/'docs/common-dense-quality-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',records=records,contrasts=contrasts)))
if __name__=='__main__':main()
