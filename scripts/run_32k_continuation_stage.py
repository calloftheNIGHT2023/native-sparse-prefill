"""Run four frozen parent continuations, then fixed held-out reports. No automatic retry."""
import argparse,hashlib,io,json,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
 out=R/'results/32k-continuation-stage-v0';out.mkdir(exist_ok=False)
 pp=R/'provenance/32k-continuation-protocol.json';p=json.loads(pp.read_text());started=utc();tick=time.monotonic();jobs=[]
 def run(name,k,seed,phase,resume,selection=None):
  remain=p['max_controller_seconds']-(time.monotonic()-tick)
  assert remain>30
  parent=R/'results/32k-adaptation-stage-v0'
  cmd=[sys.executable,'-u',str(R/'scripts/run_32k_continuation.py'),'--phase',phase,'--k',str(k),'--seed',str(seed),'--lr','0.001','--output',str(out/name),'--resume',str(resume),'--preflight',str(parent/f'preflight-k{k}')]
  if selection:cmd+=['--selection',str(selection)]
  row=dict(name=name,k=k,seed=seed,phase=phase,started_utc=utc(),command=cmd);jobs.append(row);save(out/'timeline.json',jobs)
  print(json.dumps(row),flush=True)
  with (out/f'{name}.log').open('w') as f:
   proc=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
   try:code=proc.wait(timeout=min(600,remain))
   except subprocess.TimeoutExpired:proc.kill();proc.wait();code=-9
  row.update(finished_utc=utc(),returncode=code);save(out/'timeline.json',jobs)
  assert code==0,f'{name} failed: {code}'
  result=json.loads((out/name/'result.json').read_text());assert result['status']=='complete'
  return result
 try:
  for n,h in p['source_sha256'].items():assert sha(R/n)==h,n
  for key,j in p['parents'].items():
   assert sha(R/j['checkpoint'])==j['checkpoint_sha256']
   assert sha(R/j['baseline'])==j['baseline_sha256']
  assert sha(R/p['task_data_path']/'tasks.npz')==p['task_tokens_sha256']
  for k,seed in p['run_order']:
   key=f'k{k}-seed{seed}';j=p['parents'][key]
   run(key,k,seed,'train',R/j['checkpoint'])
  # Reports are fixed endpoints, timed cuts, dense64 comparators and zero-update models.
  # Neither task outcomes nor report PPL select checkpoints or training length.
  reports=[]
  def target(name,k,seed,step,path,origin):
   reports.append(dict(name=name,k=k,seed=seed,step=step,checkpoint=str(path.relative_to(R)),checkpoint_sha256=sha(path),origin=origin))
  for k,seed in p['run_order']:
   key=f'k{k}-seed{seed}'
   target(f'report-{key}-step128',k,seed,128,out/key/'checkpoint-128.pt','continuation')
  for seed in p['seeds']:
   key=f'k32-seed{seed}';cut=json.loads((out/key/'time-budget-cut.json').read_text());step=cut['compliant_step']
   target(f'report-{key}-budget',32,seed,step,out/key/f'checkpoint-{step}.pt','continuation')
   j=p['parents'][f'k0-seed{seed}'];target(f'report-k0-seed{seed}-step64',0,seed,64,R/j['checkpoint'],'parent')
  for k in [0,32]:
   seed=p['seeds'][0];j=p['parents'][f'k{k}-seed{seed}'];target(f'report-k{k}-baseline',k,seed,0,R/j['baseline'],'parent')
  lock=out/'report-lock.json';save(lock,dict(created_utc=utc(),protocol_sha256=sha(pp),reports=reports,rule=p['report_rule']))
  for t in reports:run(t['name'],t['k'],t['seed'],'report',R/t['checkpoint'],lock)
  status='complete'
 except Exception:
  status='failed';(out/'error.txt').write_text(traceback.format_exc())
 save(out/'result.json',dict(status=status,started_utc=started,finished_utc=utc(),seconds=time.monotonic()-tick,jobs=jobs,protocol_sha256=sha(pp)))
 archive=R/'exports/32k-continuation-evidence-v0.tar.gz';assert not archive.exists()
 files=list(out.rglob('*'))+[R/n for n in p['source_sha256']]+list((R/p['task_data_path']).glob('*'))+[pp,R/'data/32k-adaptation-v0/config.json',R/'data/32k-separator-replay-v0/report.npz']
 entries=[]
 with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
  for f in sorted(set(files)):
   if not f.is_file():continue
   raw=f.read_bytes();name=f.relative_to(R).as_posix();entries.append(dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
   info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries),indent=2).encode();info=tarfile.TarInfo('32k-continuation-manifest.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 save(archive.with_suffix('.json'),dict(sha256=sha(archive),bytes=archive.stat().st_size,files=len(entries)))
 print(json.dumps(dict(status=status,archive=str(archive),seconds=time.monotonic()-tick)),flush=True)
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
