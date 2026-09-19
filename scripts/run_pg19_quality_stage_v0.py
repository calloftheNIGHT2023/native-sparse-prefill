"""Bounded evaluation only; fail closed on endpoint replay before new predictions."""
from pathlib import Path
import subprocess,sys,time,json,hashlib,tarfile,io,traceback
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    pp=R/'provenance/pg19-quality-protocol-v0.json';p=json.loads(pp.read_text())
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(R/n)==h,n
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    out=R/'results/pg19-quality-stage-v0';out.mkdir(exist_ok=False);state=R/'logs/pg19-quality-queue-v0.json';start=utc();tick=time.monotonic();jobs=[];replayed=set()
    save(state,dict(status='running',utc=start,optimizer_updates=0))
    try:
        for j in p['jobs']:
            remaining=p['maximum_seconds']-(time.monotonic()-tick);assert remaining>10
            cmd=[sys.executable,'-u',str(R/'scripts/eval_pg19_quality_v0.py'),'--job',j['name'],'--output',str(out/j['name'])]
            row=dict(name=j['name'],phase=j['phase'],started_utc=utc(),status='running',command=cmd);jobs.append(row);save(out/'timeline.json',jobs)
            with (out/f"{j['name']}.log").open('w') as f:
                q=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
                try:rc=q.wait(timeout=min(remaining,p['maximum_job_seconds']))
                except subprocess.TimeoutExpired:q.kill();q.wait();rc=-9
            row.update(returncode=rc,status='complete' if rc==0 else 'failed',finished_utc=utc());save(out/'timeline.json',jobs);assert rc==0,j['name']
            if j['phase']=='replay':replayed.add(j['k'])
            save(state,dict(status='running',utc=utc(),last_completed=j['name'],optimizer_updates=0))
        status='complete'
    except Exception:status='failed';(out/'error.txt').write_text(traceback.format_exc())
    count=0
    for j in jobs:
        f=out/j['name']/'task-predictions.jsonl'
        if f.exists():count+=len(f.read_text().splitlines())
    if status=='complete':assert count==p['expected_task_predictions']
    gradient_count=0
    if status=='complete':assert gradient_count==p['expected_gradient_passes']
    save(out/'result.json',dict(gradient_passes=gradient_count,status=status,started_utc=start,finished_utc=utc(),seconds=time.monotonic()-tick,jobs=jobs,optimizer_updates=0,task_predictions=count,protocol_sha256=sha(pp)))
    archive=R/'exports/pg19-quality-evidence-v0.tar.gz';assert not archive.exists();entries=[]
    with tarfile.open(archive,'w:gz') as t:
        files=list(out.rglob('*'))+[pp]+[R/n for n in {**p['source_sha256'],**p['data_sha256']}]
        for f in sorted(set(files)):
            if not f.is_file():continue
            raw=f.read_bytes();n=f.relative_to(R).as_posix();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries),indent=2).encode();m=tarfile.TarInfo('pg19-quality-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(archive),bytes=archive.stat().st_size,files=len(entries));save(archive.with_suffix('.json'),proof);save(state,dict(status=status,utc=utc(),optimizer_updates=0,task_predictions=count,archive=proof))
    if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
