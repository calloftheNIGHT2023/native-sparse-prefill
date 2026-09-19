"""Paired report and cost readout only after all prespecified jobs finish."""
import argparse,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main(a):
 cfg=json.loads((R/'data/flashmoba-amp-recovery-v0/config.json').read_text());stage=json.loads((a.stage/'result.json').read_text())
 assert stage['status']=='complete' and not stage['preflight_only']
 selection=json.loads((a.stage/'selection-lock.json').read_text());rows=[];reports={};trains={}
 for k in cfg['methods']:
  lr=selection['selected_lrs'][str(k)]
  for seed in cfg['seeds']:
   report=json.loads((a.stage/f'report-k{k}-seed{seed}/result.json').read_text());assert report['status']=='complete'
   reports[k,seed]=next(x for x in report['evaluations'] if x['split']=='report')['values']
   p=a.stage/(f'cal-k{k}-lr{lr:g}' if seed==cfg['seeds'][0] else f'repeat-k{k}')
   trains[k,seed]=json.loads((p/'result.json').read_text())
 for k in [4,16]:
  for seed in cfg['seeds']:
   diffs=[a-b for a,b in zip(reports[k,seed],reports[0,seed])];assert len(diffs)==12
   ratio=trains[k,seed]['training_seconds']/trains[0,seed]['training_seconds']
   rows.append(dict(k=k,seed=seed,paired_window_nll_differences=diffs,mean_nll_gap=statistics.mean(diffs),training_time_ratio=ratio,
    exploratory_screen_pass=statistics.mean(diffs)<=cfg['quality_margin_nats'] and ratio<=cfg['time_ratio_screen']))
 spend={}
 for k in cfg['methods']:
  rs=[json.loads((a.stage/f'cal-k{k}-lr{lr:g}/result.json').read_text()) for lr in cfg['learning_rates']]
  rs.append(trains[k,cfg['seeds'][1]])
  spend[str(k)]=dict(training_seconds_including_lr_search=sum(x['training_seconds'] for x in rs),process_wall_seconds_including_lr_search=sum(x['wall_seconds'] for x in rs))
 result=dict(status='complete',paired_results=rows,method_cost=spend,controller_wall_seconds=stage['seconds'],quality_margin_nats=cfg['quality_margin_nats'],scope='Exploratory two-initialization-seed stage-held-out report, not significance or novelty proof. NLL gap and time screen do not establish global convergence, equal-quality time-to-target, or native sparse pretraining.')
 p=a.stage/'paired-analysis.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);main(p.parse_args())
