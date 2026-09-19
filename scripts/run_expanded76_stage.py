"""Serial bounded expanded-data pilot, immutable evidence and no automatic retry."""
import json,hashlib,subprocess,sys,time,traceback,tarfile,io
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 pp=R/'provenance/expanded76-protocol.json';p=json.loads(pp.read_text());cfg=json.loads((R/'data/32k-expanded-training-v0/config.json').read_text())
 for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(R/n)==h,n
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 out=R/'results/expanded76-stage-v0';out.mkdir(exist_ok=False);jobs=[];tick=time.monotonic();started=utc();state=R/'logs/expanded76-queue-v0.json'
 save(state,dict(status='running',utc=utc(),protocol_sha256=sha(pp),output=str(out)))
 def run(name,phase,k,seed,resume=None):
  remain=p['maximum_seconds']-(time.monotonic()-tick);assert remain>30
  cmd=[sys.executable,'-u',str(R/'scripts/run_expanded76.py'),'--phase',phase,'--output',str(out/name),'--k',str(k),'--seed',str(seed),'--lr','0.001']
  if phase!='preflight':cmd+=['--preflight',str(out/f'preflight-k{k}')]
  if resume:cmd+=['--resume',str(resume),'--selection',str(out/'selection.json')]
  j=dict(name=name,phase=phase,k=k,seed=seed,started_utc=utc(),status='running',command=cmd);jobs.append(j);save(out/'timeline.json',jobs)
  with (out/f'{name}.log').open('w') as f:
   proc=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
   try:code=proc.wait(timeout=min(p['maximum_job_seconds'],remain))
   except subprocess.TimeoutExpired:proc.kill();proc.wait();code=-9
  j.update(returncode=code,finished_utc=utc(),status='complete' if code==0 else 'failed');save(out/'timeline.json',jobs)
  assert code==0,(name,code)
  assert json.loads((out/name/'result.json').read_text())['status']=='complete'
 try:
  for k in p['methods']:run(f'preflight-k{k}','preflight',k,p['seeds'][0])
  for k,seed in p['run_order']:run(f'k{k}-seed{seed}','train',k,seed)
  save(out/'selection.json',dict(config_sha256=sha(R/'data/32k-expanded-training-v0/config.json'),sources=cfg['sources_sha256'],selected_lrs={'0':.001,'32':.001},rule=cfg['selection_rule']))
  for k,seed in p['run_order']:run(f'report-k{k}-seed{seed}-step128','report',k,seed,out/f'k{k}-seed{seed}/checkpoint-128.pt')
  for k in p['methods']:
   seed=p['seeds'][0];run(f'report-k{k}-seed{seed}-step0','report',k,seed,out/f'k{k}-seed{seed}/checkpoint-0.pt')
  status='complete'
 except Exception:
  status='failed';(out/'error.txt').write_text(traceback.format_exc())
 counts={'scientific_updates':0,'diagnostic_updates':0,'task_predictions':0};unknown=[]
 for j in jobs:
  f=out/j['name']/'result.json'
  if not f.exists():unknown.append(j['name']);continue
  v=json.loads(f.read_text());key='diagnostic_updates' if j['phase']=='preflight' else 'scientific_updates';counts[key]+=v.get('optimizer_updates_this_process',0)
  f=out/j['name']/'task-predictions.jsonl'
  if f.exists():counts['task_predictions']+=len(f.read_text().splitlines())
 save(out/'result.json',dict(status=status,started_utc=started,finished_utc=utc(),seconds=time.monotonic()-tick,jobs=jobs,protocol_sha256=sha(pp),counts=counts,incomplete_jobs=unknown))
 a=R/'exports/expanded76-evidence-v0.tar.gz';assert not a.exists();entries=[]
 with tarfile.open(a,'w:gz',compresslevel=1) as t:
  files=list(out.rglob('*'))+[R/n for n in p['source_sha256']]+[R/n for n in p['data_sha256']]+[pp]
  for f in sorted(set(files)):
   if not f.is_file():continue
   raw=f.read_bytes();name=f.relative_to(R).as_posix();entries.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));m=tarfile.TarInfo(name);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries),indent=2).encode();m=tarfile.TarInfo('expanded76-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 save(a.with_suffix('.json'),dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries)))
 save(state,dict(status=status,utc=utc(),counts=counts,archive_sha256=sha(a)))
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':
 try:main()
 except Exception:
  save(R/'logs/expanded76-queue-v0.json',dict(status='failed',utc=utc(),error=traceback.format_exc()));raise
