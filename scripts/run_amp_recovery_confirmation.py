"""One replication followed by locked evaluation; no outcome-driven additions."""
import json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def main():
 out=R/'results/amp-recovery-stage-v0';out.mkdir(parents=True,exist_ok=False);started=utc();tick=time.monotonic();runs=[];error=None
 def call(name,cmd):
  remaining=1500-(time.monotonic()-tick)
  if remaining<30:raise TimeoutError('Stage wall limit reached')
  row=dict(name=name,started_utc=utc(),command=cmd);print(json.dumps(row),flush=True)
  with (out/f'{name}.log').open('w') as f:
   p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
   try:code=p.wait(timeout=min(750,remaining))
   except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
  row.update(finished_utc=utc(),returncode=code);runs.append(row);(out/'timeline.json').write_text(json.dumps(runs,indent=2)+'\n')
  if code:raise RuntimeError(f'{name} failed; no retry')
 try:
  for phase,name in [('preflight','preflight-k16'),('train','repeat-k16')]:
   cmd=[sys.executable,'-u',str(R/'scripts/run_amp_recovery.py'),'--phase',phase,'--output',str(out/name),'--k','16','--seed','2026091561','--lr','0.001']
   if phase=='train':cmd+=['--preflight',str(out/'preflight-k16')]
   call(name,cmd)
  parents=[(0,2026091560,R/'parents/dense-seed0'),(0,2026091561,R/'parents/dense-seed1'),(16,2026091560,R/'parents/k16-seed0'),(16,2026091561,out/'repeat-k16')]
  for k,seed,parent in parents:
   name=f'heldout-k{k}-seed{seed}';call(name,[sys.executable,'-u',str(R/'scripts/eval_amp_recovery_confirmation.py'),'--run-dir',str(parent),'--output',str(out/name)])
  status='complete'
 except Exception as e:status='failed';error=str(e)
 (out/'result.json').write_text(json.dumps(dict(status=status,error=error,started_utc=started,finished_utc=utc(),seconds=time.monotonic()-tick,runs=runs),indent=2)+'\n')
 (R/'exports').mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(R/'scripts/package_amp_recovery_results.py'),'--archive',str(R/'exports/amp-confirmation-evidence-v0.tar.gz')],check=True,timeout=180)
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
