"""Run one bounded cost screen and package its exact evidence."""
import json,subprocess,sys,time,tarfile,io,hashlib
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
out=R/'results/task-density-cost-stage-v0';out.mkdir(exist_ok=False)
start=utc();tick=time.monotonic();error=None
cmd=[sys.executable,'-u',str(R/'scripts/benchmark_task_density_cost.py'),'--run-dir',str(R/'parents/dense-seed0'),'--output',str(out/'benchmark')]
with (out/'benchmark.log').open('w') as f:
 p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
 try:code=p.wait(timeout=180)
 except subprocess.TimeoutExpired:p.kill();p.wait();code=-9;error='180-second time cap'
result=dict(status='complete' if code==0 else 'failed',error=error,returncode=code,started_utc=start,finished_utc=utc(),seconds=time.monotonic()-tick,command=cmd,optimizer_updates=0)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
files=[p for p in out.rglob('*') if p.is_file()]
files += [R/'scripts/benchmark_task_density_cost.py',Path(__file__),R/'provenance/task-quality-density-cost-protocol.json']
a=R/'exports/task-density-cost-evidence-v0.tar.gz';assert not a.exists();rows=[]
with tarfile.open(a,'w:gz') as t:
 for p in sorted(files):
  raw=p.read_bytes();name=p.relative_to(R).as_posix();rows.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));m=tarfile.TarInfo(name);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 raw=json.dumps(dict(files=rows,completed_job_directories=['results/task-density-cost-stage-v0/benchmark'] if code==0 else []),indent=2).encode();m=tarfile.TarInfo('amp-recovery-results-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
proof=dict(sha256=hashlib.sha256(a.read_bytes()).hexdigest(),bytes=a.stat().st_size,files=len(rows));a.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
raise SystemExit(0 if code==0 else 1)
