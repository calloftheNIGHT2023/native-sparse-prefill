"""Read-only checkpoint evaluation; bounded runtime and archived evidence."""
import hashlib,io,json,subprocess,sys,tarfile,time
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def main():
 pp=R/'provenance/checkpoint-quality-cost-protocol.json';protocol=json.loads(pp.read_text())
 for name,h in protocol['sources'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
 out=R/'results/checkpoint-quality-cost-stage-v0';out.mkdir(exist_ok=False)
 start=utc();tick=time.monotonic();runs=[];error=None
 try:
  for name in protocol['job_order']:
   spec=protocol['runs'][name];remaining=protocol['max_seconds']-(time.monotonic()-tick)
   if remaining<20:raise TimeoutError('Total runtime limit')
   cmd=[sys.executable,'-u',str(R/'scripts/eval_checkpoint_quality_cost.py'),'--run-dir',str(R/spec['remote_run_dir']),'--name',name,'--output',str(out/name)]
   row=dict(name=name,started_utc=utc(),command=cmd,timeout_seconds=min(protocol['per_job_max_seconds'],remaining))
   with (out/f'{name}.log').open('w') as f:
    p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
    try:code=p.wait(timeout=row['timeout_seconds'])
    except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
   row.update(returncode=code,finished_utc=utc());runs.append(row)
   (out/'timeline.json').write_text(json.dumps(runs,indent=2)+'\n');print(json.dumps(row),flush=True)
   if code:raise RuntimeError(name+' failed')
  status='complete'
 except Exception as exc:status='failed';error=repr(exc)
 result=dict(status=status,error=error,started_utc=start,finished_utc=utc(),seconds=time.monotonic()-tick,runs=runs,optimizer_updates=0,
             protocol_sha256=hashlib.sha256(pp.read_bytes()).hexdigest())
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 files=[p for p in out.rglob('*') if p.is_file()]+[R/n for n in protocol['sources']]+[pp,R/'docs/checkpoint-quality-cost-protocol-2026-09-16.md']
 archive=R/'exports/checkpoint-quality-cost-evidence-v0.tar.gz';assert not archive.exists();entries=[]
 with tarfile.open(archive,'w:gz') as tar:
  for path in sorted(set(files)):
   raw=path.read_bytes();name=path.relative_to(R).as_posix();entries.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
   member=tarfile.TarInfo(name);member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries,completed_job_directories=[f'results/checkpoint-quality-cost-stage-v0/{x["name"]}' for x in runs if x['returncode']==0]),indent=2).encode()
  member=tarfile.TarInfo('amp-recovery-results-manifest.json');member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
 proof=dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(entries))
 archive.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(dict(result=result,archive=proof)),flush=True)
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
