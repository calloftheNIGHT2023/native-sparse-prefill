"""Two sparse trajectories with verified reused dense controls; bounded and serial."""
import json,hashlib,subprocess,sys,time,traceback,tarfile,io
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 pp=R/'provenance/gentle32k-protocol.json';p=json.loads(pp.read_text());cfg=json.loads((R/'data/gentle32k-v0/config.json').read_text());k=p['k'];state=R/'logs/gentle32k-queue-v0.json'
 for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(R/n)==h,n
 assert sha(R/'data/gentle32k-v0/config.json')==p['config_sha256']
 parent=R/'results/expanded76-stage-v0';assert json.loads((parent/'result.json').read_text())['status']=='complete'
 assert sha(R/'data/32k-expanded-training-v0/config.json')==p['parent_config_sha256']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 out=R/'results/gentle32k-stage-v0';out.mkdir(exist_ok=False);tick=time.monotonic();started=utc();jobs=[];save(state,dict(status='running',utc=utc(),chosen=p['chosen']))
 def run(name,phase,seed,resume=None):
  remain=p['maximum_seconds']-(time.monotonic()-tick);assert remain>30
  cmd=[sys.executable,'-u',str(R/'scripts/run_gentle32k.py'),'--phase',phase,'--output',str(out/name),'--k',str(k),'--seed',str(seed),'--lr','0.001']
  if phase!='preflight':cmd+=['--preflight',str(out/'preflight')]
  if resume:cmd+=['--resume',str(resume),'--selection',str(out/'selection.json')]
  j=dict(name=name,phase=phase,k=k,seed=seed,started_utc=utc(),status='running',command=cmd);jobs.append(j);save(out/'timeline.json',jobs)
  with (out/f'{name}.log').open('w') as f:
   proc=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
   try:code=proc.wait(timeout=min(p['maximum_job_seconds'],remain))
   except subprocess.TimeoutExpired:proc.kill();proc.wait();code=-9
  j.update(status='complete' if code==0 else 'failed',returncode=code,finished_utc=utc());save(out/'timeline.json',jobs);assert code==0,(name,code)
 try:
  # Dense-reference replay only; checkpoint optimizer is not advanced.
  for seed in p['seeds']:
   cmd=[sys.executable,str(R/'scripts/replay_gentle_dense_reference.py'),'--seed',str(seed),'--output',str(out/f'dense-replay-{seed}')]
   with (out/f'dense-replay-{seed}.log').open('w') as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
  run('preflight','preflight',p['seeds'][0])
  for seed in p['seeds']:run(f'seed{seed}','train',seed)
  save(out/'selection.json',dict(config_sha256=p['config_sha256'],sources=cfg['sources_sha256'],selected_lrs={str(k):.001},rule=cfg['selection_rule']))
  for seed in p['seeds']:run(f'report-seed{seed}-step128','report',seed,out/f'seed{seed}/checkpoint-128.pt')
  seed=p['seeds'][0];run(f'report-seed{seed}-step0','report',seed,out/f'seed{seed}/checkpoint-0.pt')
  status='complete'
 except Exception:status='failed';(out/'error.txt').write_text(traceback.format_exc())
 counts=dict(scientific_updates=0,diagnostic_updates=0,task_predictions=0);incomplete=[]
 for j in jobs:
  f=out/j['name']/'result.json'
  if not f.exists():incomplete.append(j['name']);continue
  v=json.loads(f.read_text());counts['diagnostic_updates' if j['phase']=='preflight' else 'scientific_updates']+=v.get('optimizer_updates_this_process',0)
  f=out/j['name']/'task-predictions.jsonl'
  if f.exists():counts['task_predictions']+=len(f.read_text().splitlines())
 save(out/'result.json',dict(status=status,started_utc=started,finished_utc=utc(),seconds=time.monotonic()-tick,jobs=jobs,counts=counts,incomplete_jobs=incomplete,protocol_sha256=sha(pp)))
 a=R/'exports/gentle32k-evidence-v0.tar.gz';assert not a.exists();entries=[]
 with tarfile.open(a,'w:gz',compresslevel=1) as t:
  files=list(out.rglob('*'))+[pp,R/'data/gentle32k-v0/config.json']+[R/n for n in p['source_sha256']]+[R/n for n in p['data_sha256']]
  for f in sorted(set(files)):
   if not f.is_file():continue
   raw=f.read_bytes();n=f.relative_to(R).as_posix();entries.append(dict(path=n,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries),indent=2).encode();m=tarfile.TarInfo('gentle32k-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 save(a.with_suffix('.json'),dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries)));save(state,dict(status=status,utc=utc(),counts=counts,archive_sha256=sha(a)))
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
