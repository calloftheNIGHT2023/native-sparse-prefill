"""Finite gated batch after controlled compile continuation; preserves original batch v0."""
import hashlib,json,subprocess,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/flashmoba-batch-v1';OUT.mkdir(parents=True,exist_ok=False)
PY=ROOT/'.venv-flashmoba-v0/bin/python';start=datetime.now(timezone.utc).isoformat();tick=time.perf_counter();stages=[]
def utc():return datetime.now(timezone.utc).isoformat()
def event(kind,**kw):
    row=dict(event=kind,utc=utc(),elapsed_seconds=time.perf_counter()-tick,**kw)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def run(name,script,args,timeout=900):
    cmd=[str(PY),'scripts/'+script]+args;begin=utc();t=time.perf_counter()
    event('stage_start',name=name,command=cmd,script_sha256=hashlib.sha256((ROOT/'scripts'/script).read_bytes()).hexdigest())
    with (ROOT/'logs'/f'{name}.log').open('x') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    row=dict(name=name,started_utc=begin,finished_utc=utc(),wall_seconds=time.perf_counter()-t,exit_code=r.returncode)
    stages.append(row);event('stage_complete',**row);assert r.returncode==0,f'{name} failed; stopped'
try:
    event('waiting_for_environment_resume')
    env=ROOT/'results/flashmoba-environment-resume-v1/environment.json'
    while not env.exists():
        assert time.perf_counter()-tick<3000;time.sleep(10)
    assert json.loads(env.read_text())['status']=='complete'
    original=ROOT/'results/flashmoba-batch-v0/batch.json'
    assert original.exists() and json.loads(original.read_text())['status']=='incomplete'
    original_gate='results/flashmoba-official-gate-v0/verification.json'
    metadata_gate='results/flashmoba-metadata-gate-v0/verification.json'
    for name,script,extra in [
        ('flashmoba-official-gate-v0','verify_flashmoba_official.py',[]),
        ('flashmoba-official-prefill-v0','benchmark_flashmoba_official.py',['--gate',original_gate]),
        ('flashmoba-metadata-gate-v0','verify_flashmoba_metadata.py',['--gate',original_gate]),
        ('flashmoba-fixedmeta-prefill-v0','benchmark_flashmoba_graphs.py',['--gate',metadata_gate]),
        ('flashmoba-training-core-v0','benchmark_flashmoba_training_core.py',['--gate',original_gate])]:
        run(name,script,['--output','results/'+name]+extra)
    run('flashmoba-wheel-package-v0','package_flashmoba_wheel.py',['--environment-record',str(env)],timeout=650)
    status='complete';error=None
except Exception as exc:
    status='incomplete';error=dict(message=str(exc),traceback=traceback.format_exc());event('failed',error=error)
(OUT/'batch.json').write_text(json.dumps(dict(status=status,error=error,stages=stages,started_utc=start,finished_utc=utc(),
    wall_seconds=time.perf_counter()-tick,scientific_optimizer_updates=0),indent=2))
if status!='complete':raise SystemExit(1)
