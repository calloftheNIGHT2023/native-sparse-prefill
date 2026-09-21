"""Audit pairing and summarize the fixed ten-run pilot without choosing new settings."""
import hashlib,json,math,statistics
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-quality-cost-analysis-v0';OUT.mkdir(parents=True,exist_ok=False)
cfg=json.loads((ROOT/'data/flashmoba-quality-cost-v0/config.json').read_text())
controller=json.loads((ROOT/'results/flashmoba-quality-cost-controller-v1/result.json').read_text())
assert controller['status']=='complete' and controller['scientific_optimizer_updates']==640
assert hashlib.sha256((ROOT/'data/flashmoba-quality-cost-v0/tokens.npz').read_bytes()).hexdigest()==cfg['tokens_sha256']
results={};summary=[];pairs=[]
for seed in cfg['initialization_seeds']:
    initial=set()
    for cond in cfg['conditions']:
        p=ROOT/'results'/f"flashmoba-quality-cost-{cond['name']}-s{seed}-v1/result.json"
        x=json.loads(p.read_text());assert x['status']=='complete' and x['scientific_optimizer_updates']==64
        assert x['seed']==seed and x['condition']==cond and x['tokens_sha256']==cfg['tokens_sha256']
        assert x['extension_sha256']==cfg['extension_sha256'][cond['extension']]
        initial.add(x['initial_adapters_tensor_sha256']);results[(seed,cond['name'])]=x
    assert len(initial)==1,'Conditions did not start from the same adapter tensor bytes.'
    dense=results[(seed,'dense')];dense_report=next(e['nlls'] for e in dense['evaluations'] if e['split']=='report')
    for cond in cfg['conditions']:
        x=results[(seed,cond['name'])];cal=[e for e in x['evaluations'] if e['split']=='calibration']
        final=next(e['nlls'] for e in x['evaluations'] if e['split']=='report')
        assert len(final)==16 and [e['step'] for e in cal]==[0,16,32,64]
        delta=np.asarray(final)-dense_report
        ratio=x['steady_step_seconds_median']/dense['steady_step_seconds_median']
        k=cond['topk'];n=cfg['length'];b=cfg['block_size']
        edges=sum(min(t//b,k-1)*b+t%b+1 for t in range(n)) if k else n*(n+1)//2
        row=dict(seed=seed,condition=cond['name'],calibration_initial=cal[0]['mean_nll'],calibration_final=cal[-1]['mean_nll'],
            report_nll=float(np.mean(final)),paired_report_gap=float(delta.mean()),paired_report_deltas=delta.tolist(),
            report_perplexity_ratio=math.exp(float(delta.mean())),steady_step_seconds_median=x['steady_step_seconds_median'],
            steady_step_time_ratio=ratio,training_seconds=x['train_step_seconds_total'],full_job_seconds=x['wall_seconds'],
            peak_gpu_gib=x['peak_gpu_bytes']/2**30,logical_causal_edge_fraction=edges/(n*(n+1)/2),
            quality_screen_pass=float(delta.mean())<=cfg['quality_screen_margin_nats'],cost_screen_pass=ratio<=cfg['training_time_ratio_screen'],
            calibration_curve=[dict(step=e['step'],mean_nll=e['mean_nll']) for e in cal])
        summary.append(row)
    for left,right in [('original_k4','barrier_k4'),('barrier_k4','fp32_k4'),('fp32_k4','fp32_k16')]:
        a=results[(seed,left)];z=results[(seed,right)]
        av=np.asarray(next(e['nlls'] for e in a['evaluations'] if e['split']=='report'))
        zv=np.asarray(next(e['nlls'] for e in z['evaluations'] if e['split']=='report'))
        pairs.append(dict(seed=seed,left=left,right=right,paired_report_gap=float((zv-av).mean()),
            report_deltas=(zv-av).tolist(),time_ratio=z['steady_step_seconds_median']/a['steady_step_seconds_median'],
            same_final_delta_sha256=a['final_delta_sha256']==z['final_delta_sha256']))
assert results[(cfg['initialization_seeds'][0],'dense')]['initial_adapters_tensor_sha256']!=results[(cfg['initialization_seeds'][1],'dense')]['initial_adapters_tensor_sha256']
aggregate=[]
for cond in cfg['conditions']:
    selected=[x for x in summary if x['condition']==cond['name']]
    # Report windows repeat across seeds; do not count the 32 values as independent documents.
    gap=np.mean([x['paired_report_deltas'] for x in selected],axis=0)
    bootstrap=np.random.default_rng(2026091554).choice(gap,(10000,len(gap)),replace=True).mean(axis=1)
    aggregate.append(dict(condition=cond['name'],mean_report_nll=float(np.mean([x['report_nll'] for x in selected])),
        mean_paired_gap=float(gap.mean()),paired_window_bootstrap_95pct=np.quantile(bootstrap,[.025,.975]).tolist(),
        report_gap_seed_range=[min(x['paired_report_gap'] for x in selected),max(x['paired_report_gap'] for x in selected)],
        training_time_ratio_seed_range=[min(x['steady_step_time_ratio'] for x in selected),max(x['steady_step_time_ratio'] for x in selected)],
        both_seeds_quality_pass=all(x['quality_screen_pass'] for x in selected),both_seeds_cost_pass=all(x['cost_screen_pass'] for x in selected),
        pass_expansion_screen=all(x['quality_screen_pass'] and x['cost_screen_pass'] for x in selected),
        uncertainty_scope='Exploratory paired resampling of 16 windows after averaging seeds; adjacent/correlated WikiText windows, not 32 independent documents or a population noninferiority test.'))
result=dict(status='complete',analyzed_utc=datetime.now(timezone.utc).isoformat(),rows=summary,pairs=pairs,aggregate=aggregate,
    optimizer_updates=640,trajectories=10,paired_initialization_verified=True,paired_tokens_verified=True,
    report_windows=16,initialization_seeds=cfg['initialization_seeds'],any_sparse_passes_expansion_screen=any(x['pass_expansion_screen'] for x in aggregate if x['condition']!='dense'),
    scope=cfg['scope'],gate={'nll_margin_nats':cfg['quality_screen_margin_nats'],'training_time_ratio':cfg['training_time_ratio_screen']})
(OUT/'source.py').write_bytes(Path(__file__).read_bytes());(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['rows','pairs']},indent=2),flush=True)
