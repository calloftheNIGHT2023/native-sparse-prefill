"""Pilot capability gate then locked task evaluation; never optimize weights."""
import subprocess,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def main():
 out=R/'results/task-quality-stage-v0';out.mkdir(parents=True,exist_ok=False);tick=time.monotonic();started=utc();runs=[];error=None
 def run(name,parent,split,gate=None):
  remaining=1200-(time.monotonic()-tick)
  if remaining<30:raise TimeoutError('Task stage time cap')
  cmd=[sys.executable,'-u',str(R/'scripts/eval_amp_recovery_tasks.py'),'--run-dir',str(parent),'--output',str(out/name),'--split',split]
  if gate:cmd+=['--gate',str(gate)]
  row=dict(name=name,started_utc=utc(),command=cmd);print(json.dumps(row),flush=True)
  with (out/f'{name}.log').open('w') as f:
   proc=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
   try:code=proc.wait(timeout=min(360,remaining))
   except subprocess.TimeoutExpired:proc.kill();proc.wait();code=-9
  row.update(finished_utc=utc(),returncode=code);runs.append(row);(out/'timeline.json').write_text(json.dumps(runs,indent=2)+'\n')
  if code:raise RuntimeError(f'{name} failed, no repeat')
 try:
  run('pilot-dense0',R/'parents/dense-seed0','pilot');gate=out/'pilot-dense0/gate.json';decision=json.loads(gate.read_text())
  if decision['selected_tasks']:
   # Same four fixed final checkpoints, reverse order of dense/sparse for seed1.
   for name,parent in [('dense0',R/'parents/dense-seed0'),('sparse0',R/'parents/k16-seed0'),('sparse1',R/'results/amp-recovery-stage-v0/repeat-k16'),('dense1',R/'parents/dense-seed1')]:run(name,parent,'formal',gate)
   status='complete'
  else:status='capability_gate_failed'
 except Exception as e:status='failed';error=str(e)
 (out/'result.json').write_text(json.dumps(dict(status=status,error=error,started_utc=started,finished_utc=utc(),seconds=time.monotonic()-tick,runs=runs,optimizer_updates=0),indent=2)+'\n')
 print(json.dumps(dict(status=status,seconds=time.monotonic()-tick)),flush=True)
 subprocess.run([sys.executable,str(R/'scripts/package_task_quality.py')],check=True,timeout=180)
 if status=='failed':raise SystemExit(1)
if __name__=='__main__':main()
