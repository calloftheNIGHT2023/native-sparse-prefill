"""One additional LR for dense and K16; calibration only, bounded, no retries."""
import json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def main():
 out=R/'results/amp-recovery-stage-v0';out.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((R/'data/flashmoba-amp-recovery-v0/config.json').read_text())
 assert cfg['methods']==[0,16] and cfg['learning_rates']==[.001] and cfg['steps']==256
 assert not (R/'data/flashmoba-amp-recovery-v0/report.npz').exists()
 started=utc();tick=time.monotonic();runs=[];status='running';error=None
 def run(name,phase,k):
  remaining=1500-(time.monotonic()-tick)
  if remaining<30:raise RuntimeError('Controller time cap reached')
  cmd=[sys.executable,'-u',str(R/'scripts/run_amp_recovery.py'),'--phase',phase,'--output',str(out/name),'--k',str(k),'--seed',str(cfg['seeds'][0]),'--lr','0.001']
  if phase=='train':cmd+=['--preflight',str(out/f'preflight-k{k}')]
  row=dict(name=name,started_utc=utc(),command=cmd);print(json.dumps(row),flush=True)
  with (out/f'{name}.log').open('w') as f:
   p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
   try:code=p.wait(timeout=min(750,remaining))
   except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
  row.update(finished_utc=utc(),returncode=code);runs.append(row)
  (out/'timeline.json').write_text(json.dumps(runs,indent=2)+'\n')
  if code:raise RuntimeError(f'{name} failed with code {code}; no retry')
 try:
  for k in [0,16]:run(f'preflight-k{k}','preflight',k)
  for k in [0,16]:run(f'cal-k{k}-lr0.001','train',k)
  status='complete'
 except Exception as e:status='failed';error=str(e)
 result=dict(status=status,started_utc=started,finished_utc=utc(),seconds=time.monotonic()-tick,runs=runs,error=error,report_evaluations=0)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 (R/'exports').mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(R/'scripts/package_amp_recovery_results.py'),'--archive',str(R/'exports/amp-lr-boundary-evidence-v0.tar.gz')],check=True,timeout=180)
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
