"""Complete four locked evaluations from saved checkpoints; never train."""
import json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def main():
 out=R/'results/amp-recovery-stage-v1';out.mkdir(parents=True,exist_ok=False);tick=time.monotonic();started=utc();runs=[];error=None
 parents=[(0,2026091560,R/'parents/dense-seed0'),(0,2026091561,R/'parents/dense-seed1'),(16,2026091560,R/'parents/k16-seed0'),(16,2026091561,R/'results/amp-recovery-stage-v0/repeat-k16')]
 try:
  for k,seed,parent in parents:
   remaining=600-(time.monotonic()-tick)
   if remaining<20:raise TimeoutError('Evaluation controller wall limit')
   name=f'heldout-k{k}-seed{seed}'
   cmd=[sys.executable,'-u',str(R/'scripts/eval_amp_recovery_confirmation_v1.py'),'--run-dir',str(parent),'--output',str(out/name)]
   row=dict(name=name,started_utc=utc(),command=cmd);print(json.dumps(row),flush=True)
   with (out/f'{name}.log').open('w') as f:
    p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
    try:code=p.wait(timeout=min(180,remaining))
    except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
   row.update(finished_utc=utc(),returncode=code);runs.append(row);(out/'timeline.json').write_text(json.dumps(runs,indent=2)+'\n')
   if code:raise RuntimeError(f'{name} failed; preserve evidence')
  status='complete'
 except Exception as e:status='failed';error=str(e)
 (out/'result.json').write_text(json.dumps(dict(status=status,error=error,started_utc=started,finished_utc=utc(),seconds=time.monotonic()-tick,runs=runs,optimizer_updates=0),indent=2)+'\n')
 subprocess.run([sys.executable,str(R/'scripts/package_amp_recovery_results.py'),'--archive',str(R/'exports/amp-confirmation-evidence-v1.tar.gz')],check=True,timeout=180)
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
