"""Pair timing blocks within precision, length and round; no quality claims."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-mixed-cost-analysis-v2';OUT.mkdir(parents=True,exist_ok=False)
cfg=json.loads((ROOT/'configs/flashmoba-mixed-cost-v2.json').read_text())
controller=json.loads((ROOT/'results/flashmoba-mixed-cost-controller-v2/result.json').read_text())
rows=[];gates=[];failed=[];aggregate=[]
for precision,length in cfg['configurations']:
    path=ROOT/'results'/f'flashmoba-mixed-cost-{precision}-{length}-v2/result.json'
    if not path.exists():
        failed.append(dict(precision=precision,length=length,status='not_run'));continue
    result=json.loads(path.read_text())
    gates.append(dict(precision=precision,length=length,gates=result['gates']))
    if result['status']!='complete':
        failed.append(dict(precision=precision,length=length,status=result['status'],error=result.get('error')));continue
    assert result['scientific_optimizer_updates']==0 and result['adapter_parameters_unchanged']
    assert len(result['rows'])==6 and all(g['passed'] for g in result['gates'])
    for r in result['rows']:
        assert len(r['samples'])==cfg['samples']
        dense=next(d for d in result['rows'] if d['k']==0 and d['round']==r['round'] and d['mode']==r['mode'])
        rows.append(dict(precision=precision,length=length,k=r['k'],mode=r['mode'],round=r['round'],
                         median_wall_seconds=r['median_wall_seconds'],
                         paired_time_ratio=r['median_wall_seconds']/dense['median_wall_seconds'],
                         tokens_per_second=length/r['median_wall_seconds'],peak_gpu_gib=r['peak_gpu_bytes']/2**30,
                         samples=r['samples']))
    for mode in ['prefill']:
        for k in cfg['conditions_topk']:
            selected=[r for r in rows if r['precision']==precision and r['length']==length and r['mode']==mode and r['k']==k]
            ratios=[r['paired_time_ratio'] for r in selected]
            aggregate.append(dict(precision=precision,length=length,mode=mode,k=k,
                mean_round_median_seconds=float(np.mean([r['median_wall_seconds'] for r in selected])),
                paired_time_ratio_range=[min(ratios),max(ratios)],
                peak_gpu_gib_range=[min(r['peak_gpu_gib'] for r in selected),max(r['peak_gpu_gib'] for r in selected)],
                both_rounds_cost_screen_pass=(k!=0 and max(ratios)<=cfg['screen_max_paired_time_ratio'])))
result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),controller_status=controller['status'],
            rows=rows,aggregate=aggregate,gates=gates,failed_or_not_run=failed,scientific_optimizer_updates=0,
            measured_calls=sum(len(r['samples']) for r in rows),timing_blocks=len(rows),scope=cfg['scope'],
            interpretation='Two rounds are repeated timing blocks, not independent training seeds; no accuracy or quality recovery was measured.')
(OUT/'source.py').write_bytes(Path(__file__).read_bytes());(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
