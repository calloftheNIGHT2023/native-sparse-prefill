"""Sequential bounded GPU stage; no auto-retries, no implicit report tuning."""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
from amp_recovery_selection import select_lrs
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def main(a):
 cfg=json.loads((R/'data/32k-adaptation-v0/config.json').read_text());out=a.output;out.mkdir(parents=True,exist_ok=False)
 started=utc();start=time.monotonic();runs=[]
 protocol_path=R/'provenance/32k-adaptation-protocol.json';protocol=json.loads(protocol_path.read_text())
 assert sha(R/'data/32k-adaptation-v0/config.json')==protocol['config_sha256']
 for n,h in protocol['source_sha256'].items():assert sha(R/n)==h
 def run(name,phase,k,seed,lr,resume=None,selection=None):
  remain=cfg['maximum_controller_seconds']-(time.monotonic()-start)
  if remain<30:raise RuntimeError('Stage wall-time limit reached; no further launch.')
  dest=out/name;cmd=[sys.executable,'-u',str(R/'scripts/run_32k_adaptation.py'),'--phase',phase,'--output',str(dest),'--k',str(k),'--seed',str(seed),'--lr',str(lr)]
  if phase!='preflight':cmd+=['--preflight',str(out/f'preflight-k{k}')]
  if resume:cmd+=['--resume',str(resume)]
  if selection:cmd+=['--selection',str(selection)]
  row=dict(name=name,started_utc=utc(),command=cmd);print(json.dumps(row),flush=True)
  with (out/f'{name}.log').open('w') as f:
   p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
   try:code=p.wait(timeout=min(900,remain))
   except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
  row.update(finished_utc=utc(),returncode=code);runs.append(row)
  (out/'timeline.json').write_text(json.dumps(runs,indent=2)+'\n')
  if code:raise RuntimeError(f'{name} failed ({code}); inspect log and preserved checkpoints.')
  return json.loads((dest/'result.json').read_text())
 try:
  for k in cfg['methods']:run(f'preflight-k{k}','preflight',k,cfg['seeds'][0],cfg['learning_rates'][1])
  if not a.preflight_only:
   results=[]
   # Rotate method order across LR trials, same number of trials for every method.
   for j,lr in enumerate(cfg['learning_rates']):
    order=cfg['methods'][j:]+cfg['methods'][:j]
    for k in order:results.append(run(f'cal-k{k}-lr{lr:g}','train',k,cfg['seeds'][0],lr))
   chosen=select_lrs(results,cfg)
   selection=dict(created_utc=utc(),selected_lrs=chosen,config_sha256=results[0]['identity']['config_sha256'],sources=results[0]['identity']['sources'],calibration_results=[dict(path=(out/f"cal-k{r['identity']['k']}-lr{r['identity']['lr']:g}/result.json").as_posix(),sha256=sha(out/f"cal-k{r['identity']['k']}-lr{r['identity']['lr']:g}/result.json")) for r in results])
   selection_path=out/'selection-lock.json';selection_path.write_text(json.dumps(selection,indent=2)+'\n')
   for k in reversed(cfg['methods']):run(f'repeat-k{k}','train',k,cfg['seeds'][1],chosen[str(k)])
   # Only now open report tokens, for all methods and both seeds.
   for k in cfg['methods']:
    lr=chosen[str(k)]
    for seed in cfg['seeds']:
     parent=out/(f'cal-k{k}-lr{lr:g}' if seed==cfg['seeds'][0] else f'repeat-k{k}')
     run(f'report-k{k}-seed{seed}','report',k,seed,lr,parent/f"checkpoint-{cfg['steps']}.pt",selection_path)
   for k in cfg['methods']:
    lr=chosen[str(k)];parent=out/f'cal-k{k}-lr{lr:g}'
    run(f'baseline-k{k}','report',k,cfg['seeds'][0],lr,parent/'checkpoint-0.pt',selection_path)
  status='complete'
 except Exception as e:status='failed';(out/'error.txt').write_text(str(e)+'\n')
 (out/'result.json').write_text(json.dumps(dict(status=status,started_utc=started,finished_utc=utc(),seconds=time.monotonic()-start,runs=runs,preflight_only=a.preflight_only),indent=2)+'\n')
 import package_32k_adaptation
 package_32k_adaptation.package(out)
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');main(p.parse_args())
