"""Bounded six-cell task evaluation; no optimization or automatic retry."""
import json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def main():
 protocol=json.loads((R/'provenance/task-quality-factorial-protocol.json').read_text())
 out=R/'results/task-factorial-stage-v0';out.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();started=utc();runs=[];error=None
 try:
  for name in protocol['order']:
   remain=protocol['maximum_seconds']-(time.monotonic()-start)
   if remain<30:raise TimeoutError('Stage time cap')
   c=protocol['conditions'][name]
   cmd=[sys.executable,'-u',str(R/'scripts/eval_amp_recovery_task_factorial.py'),'--condition',name,'--run-dir',c['run_dir'],'--output',str(out/name),'--split','formal','--gate',str(R/'results/task-quality-stage-v0/pilot-dense0/gate.json')]
   row=dict(name=name,started_utc=utc(),command=cmd);print(json.dumps(row),flush=True)
   with (out/f'{name}.log').open('w') as f:
    p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
    try:code=p.wait(timeout=min(protocol['per_job_seconds'],remain))
    except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
   row.update(finished_utc=utc(),returncode=code);runs.append(row)
   (out/'timeline.json').write_text(json.dumps(runs,indent=2)+'\n')
   if code:raise RuntimeError(name+' failed; preserve outputs')
  status='complete'
 except Exception as e:status='failed';error=str(e)
 (out/'result.json').write_text(json.dumps(dict(status=status,error=error,seconds=time.monotonic()-start,started_utc=started,finished_utc=utc(),runs=runs,optimizer_updates=0),indent=2)+'\n')
 subprocess.run([sys.executable,str(R/'scripts/package_task_factorial.py')],check=True,timeout=180)
 if status=='failed':raise SystemExit(1)
if __name__=='__main__':main()
