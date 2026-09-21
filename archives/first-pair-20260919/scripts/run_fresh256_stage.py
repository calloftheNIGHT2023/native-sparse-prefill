"""Wait for verified expanded76 evidence, then evaluate six fixed checkpoints."""
import json,hashlib,subprocess,sys,time,traceback,tarfile,io
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 pp=R/'provenance/fresh256-confirmation-protocol.json';p=json.loads(pp.read_text());state=R/'logs/fresh256-queue-v0.json';start=time.monotonic()
 for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(R/n)==h,n
 save(state,dict(status='waiting_for_parent_audit',utc=utc(),protocol_sha256=sha(pp)))
 while True:
  try:
   parent=json.loads((R/'results/expanded76-audit-v0/result.json').read_text())
   if parent['status']=='verified':break
  except (FileNotFoundError,json.JSONDecodeError):pass
  assert time.monotonic()-start<3600,'Parent audit not available; do not evaluate unverified checkpoints'
  q=json.loads((R/'logs/expanded76-queue-v0.json').read_text());assert q['status']!='failed'
  time.sleep(15)
 assert parent['control']['protocol_sha256']==p['parent_protocol_sha256']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 out=R/'results/fresh256-stage-v0';out.mkdir(exist_ok=False);jobs=[];tick=time.monotonic();started=utc();save(state,dict(status='running',utc=utc(),protocol_sha256=sha(pp)))
 locked=[]
 for k,seed,step in p['models']:
  cp=R/f'results/expanded76-stage-v0/k{k}-seed{seed}/checkpoint-{step}.pt';locked.append(dict(k=k,seed=seed,step=step,path=cp.relative_to(R).as_posix(),sha256=sha(cp)))
 save(out/'checkpoint-lock.json',dict(utc=utc(),protocol_sha256=sha(pp),models=locked))
 try:
  for m in locked:
   k,seed,step=m['k'],m['seed'],m['step'];name=f'k{k}-seed{seed}-step{step}';remain=p['maximum_seconds']-(time.monotonic()-tick);assert remain>30
   cmd=[sys.executable,'-u',str(R/'scripts/eval_fresh256.py'),'--phase','report','--output',str(out/name),'--k',str(k),'--seed',str(seed),'--lr','0.001','--resume',str(R/m['path']),'--selection',str(R/'results/expanded76-stage-v0/selection.json'),'--preflight',str(R/f'results/expanded76-stage-v0/preflight-k{k}')]
   j=dict(name=name,**m,started_utc=utc(),status='running',command=cmd);jobs.append(j);save(out/'timeline.json',jobs)
   with (out/f'{name}.log').open('w') as f:
    proc=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
    try:code=proc.wait(timeout=min(600,remain))
    except subprocess.TimeoutExpired:proc.kill();proc.wait();code=-9
   j.update(returncode=code,status='complete' if code==0 else 'failed',finished_utc=utc());save(out/'timeline.json',jobs);assert code==0,(name,code)
  status='complete'
 except Exception:status='failed';(out/'error.txt').write_text(traceback.format_exc())
 predictions=sum(len(f.read_text().splitlines()) for f in out.glob('*/task-predictions.jsonl'))
 save(out/'result.json',dict(status=status,started_utc=started,finished_utc=utc(),seconds=time.monotonic()-tick,jobs=jobs,protocol_sha256=sha(pp),scientific_updates=0,diagnostic_updates=0,task_predictions=predictions))
 a=R/'exports/fresh256-evidence-v0.tar.gz';assert not a.exists();entries=[]
 with tarfile.open(a,'w:gz',compresslevel=1) as t:
  files=list(out.rglob('*'))+[R/n for n in p['source_sha256']]+[R/n for n in p['data_sha256']]+[pp]
  for f in sorted(set(files)):
   if not f.is_file():continue
   raw=f.read_bytes();name=f.relative_to(R).as_posix();entries.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));m=tarfile.TarInfo(name);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries),indent=2).encode();m=tarfile.TarInfo('fresh256-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 save(a.with_suffix('.json'),dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries)))
 save(state,dict(status=status,utc=utc(),task_predictions=predictions,archive_sha256=sha(a)))
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':
 try:main()
 except Exception:save(R/'logs/fresh256-queue-v0.json',dict(status='failed',utc=utc(),error=traceback.format_exc()));raise
