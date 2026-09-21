"""Bounded isolated build and mathematical validation, no model training."""
import json,subprocess,sys,time,traceback,hashlib,tarfile,io,os
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 wait_start=time.monotonic();state=R/'logs/topk-m128n32-queue-v0.json';save(state,dict(status='waiting_for_cost_screen',utc=utc()))
 while True:
  q=json.loads((R/'logs/m64-density-cost-queue-v0.json').read_text())
  assert q['status']!='failed'
  if q['status']=='complete' and (R/'exports/m64-density-cost-evidence-v0.tar.json').exists():break
  assert time.monotonic()-wait_start<1800
  time.sleep(15)
 out=R/'results/topk-m128n32-stage-v0';out.mkdir(exist_ok=False);state=R/'logs/topk-m128n32-queue-v0.json';jobs=[];tick=time.monotonic();started=utc()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 save(state,dict(status='running',utc=utc(),scope='Compile isolated candidate then math checks; no optimizer updates'))
 try:
  for name,script,limit in [('build','build_topk_m128n32.py',720),('math','check_topk_m128n32.py',120)]:
   row=dict(name=name,started_utc=utc(),status='running');jobs.append(row);save(out/'timeline.json',jobs)
   with (out/f'{name}.log').open('w') as f:
    proc=subprocess.Popen([sys.executable,'-u',str(R/'scripts'/script)],cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
     import os,signal
     os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=-9
   row.update(returncode=code,status='complete' if code==0 else 'failed',finished_utc=utc());save(out/'timeline.json',jobs);assert code==0,(name,code)
  status='complete'
 except Exception:status='failed';(out/'error.txt').write_text(traceback.format_exc())
 save(out/'result.json',dict(status=status,started_utc=started,finished_utc=utc(),seconds=time.monotonic()-tick,jobs=jobs,scientific_updates=0,diagnostic_optimizer_updates=0,task_predictions=0))
 src=R/'experiments/topk-m128n32-v0';files=list(out.rglob('*'))+list((R/'results/topk-m128n32-math-v0').glob('*'))+list(src.glob('*'))+list((src/'build').glob('*.so'))
 files += [R/'scripts'/n for n in ['prepare_topk_m64.py','build_topk_m128n32.py','topk_m128n32_adapter.py','check_topk_m128n32.py','run_topk_m128n32_stage.py','density_tradeoff_math_gate.py','run_flashmoba_realtext_precision.py']]
 a=R/'exports/topk-m128n32-evidence-v0.tar.gz';assert not a.exists();entries=[]
 with tarfile.open(a,'w:gz',compresslevel=1) as t:
  for f in sorted(set(files)):
   if not f.is_file():continue
   raw=f.read_bytes();n=f.relative_to(R).as_posix();entries.append(dict(path=n,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries),indent=2).encode();m=tarfile.TarInfo('topk-m128n32-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 save(a.with_suffix('.json'),dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries)));save(state,dict(status=status,utc=utc(),archive_sha256=sha(a)))
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
