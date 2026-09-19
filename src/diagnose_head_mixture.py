"""No-training development-set diagnostic of head renormalization effects."""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json,statistics,time
from pathlib import Path
import torch
from sparse_reference import make_layout,select_blocks
from indexer_calibration import make_indexer
from head_mixture import *
from trace_math import target_from_logits,prepare_trace

ROOT=Path(__file__).resolve().parents[1]


def main():
 out=ROOT/'results/head-mixture-diagnostic-v0'
 out.mkdir(exist_ok=False)
 started=datetime.now(timezone.utc).isoformat()
 torch.set_num_threads(4)
 rows=[]
 configs=[('data/realtext-v0-r1','results/realtext-v1-optimization-check'),
          ('data/realtext-calibrated-fresh-v0','results/realtext-calibrated-fresh-v0'),
          ('data/realtext-early-step10000','results/realtext-early-step10000')]
 for data_name,run_name in configs:
  data,run=ROOT/data_name,ROOT/run_name
  cfg=json.loads((run/'frozen-config.json').read_text())
  manifest=json.loads((data/'manifest.json').read_text())
  layout=make_layout([0]*cfg['sequence_length'],cfg['block_size'])
  for layer in cfg['layer_ids']:
   for seed in cfg['seeds']:
    indexer=make_indexer(512,cfg)
    indexer.load_state_dict(torch.load(run/f'layer{layer}__seed{seed}__selected_only/final-indexer.pt',weights_only=True))
    for record in manifest['examples']:
     if record['split']!='validation':continue
     tensors=torch.load(data/record['trace_path'],map_location='cpu',weights_only=True)
     trace=prepare_trace(tensors['layers'][str(layer)],layout)
     with torch.no_grad():
      support=select_blocks(indexer(trace['hidden'],layout),layout.visible_blocks,cfg['selected_blocks'])
      oracle=dense_restricted_target(trace['logits'],support,layout)
      sparse=target_from_logits(trace['logits'],support,layout)
      exact=corrected_target(trace['logits'],support,layout,exact_retained_mass(trace['logits'],support,layout))
      estimated=corrected_target(trace['logits'],support,layout,sampled_retained_mass(trace['logits'],support,layout,2,torch.Generator().manual_seed(seed+100)))
      late=slice(cfg['late_query_start'],None)
      item={'data':data_name,'layer':layer,'seed':seed,'id':record['id'],
        'sparse_target_tv':((sparse-oracle).abs().sum(-1)*.5)[late].mean().item(),
        'estimated_target_tv':((estimated-oracle).abs().sum(-1)*.5)[late].mean().item(),
        'top1_disagreement':(sparse.argmax(-1)!=oracle.argmax(-1))[late].float().mean().item(),
        'exact_correction_max_error':(exact-oracle).abs().max().item()}
      rows.append(item)
      with (out/'events.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**item})+'\n')
 group=defaultdict(list)
 for r in rows:group[(r['data'],r['layer'])].append(r)
 aggregates=[{'data':k[0],'layer':k[1],'mean':{metric:statistics.mean(r[metric] for r in rs)
    for metric in ['sparse_target_tv','estimated_target_tv','top1_disagreement','exact_correction_max_error']}} for k,rs in group.items()]
 summary={'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'rows':rows,'aggregate':aggregates,
 'scope':'No optimizer updates. Prior development paragraphs only. Oracle dense normalizers diagnose a target difference, not a deployable speedup.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(aggregates))


if __name__=='__main__':main()
