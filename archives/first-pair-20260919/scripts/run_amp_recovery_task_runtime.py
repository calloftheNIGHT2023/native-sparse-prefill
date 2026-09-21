"""Post-hoc 2x2 weight-training versus inference-attention intervention, zero updates."""
import subprocess,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def main():
 out=R/'results/task-runtime-stage-v0';out.mkdir(parents=True,exist_ok=False);tick=time.monotonic();started=utc();runs=[];error=None
 conditions=[('dense_weights_sparse_eval0',R/'parents/dense-seed0',16),('sparse_weights_dense_eval0',R/'parents/k16-seed0',0),('sparse_weights_dense_eval1',R/'results/amp-recovery-stage-v0/repeat-k16',0),('dense_weights_sparse_eval1',R/'parents/dense-seed1',16)]
 try:
  for name,parent,k in conditions:
   remain=600-(time.monotonic()-tick)
   if remain<30:raise TimeoutError('Runtime diagnostic limit')
   cmd=[sys.executable,'-u',str(R/'scripts/eval_amp_recovery_task_runtime.py'),'--run-dir',str(parent),'--output',str(out/name),'--split','formal','--gate',str(R/'results/task-quality-stage-v0/pilot-dense0/gate.json'),'--attention-k',str(k)]
   row=dict(name=name,started_utc=utc(),command=cmd);print(json.dumps(row),flush=True)
   with (out/f'{name}.log').open('w') as f:
    p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
    try:code=p.wait(timeout=min(180,remain))
    except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
   row.update(finished_utc=utc(),returncode=code);runs.append(row);(out/'timeline.json').write_text(json.dumps(runs,indent=2)+'\n')
   if code:raise RuntimeError(name+' failed')
  status='complete'
 except Exception as e:status='failed';error=str(e)
 (out/'result.json').write_text(json.dumps(dict(status=status,error=error,started_utc=started,finished_utc=utc(),seconds=time.monotonic()-tick,runs=runs,optimizer_updates=0),indent=2)+'\n')
 subprocess.run([sys.executable,str(R/'scripts/package_task_runtime.py')],check=True,timeout=180)
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
