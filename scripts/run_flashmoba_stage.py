"""Finite sequential gate/benchmark batch after the independent compiler job completes."""
import json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-batch-v0';OUT.mkdir(parents=True,exist_ok=False)
PY=ROOT/'.venv-flashmoba-v0/bin/python'
def utc():return datetime.now(timezone.utc).isoformat()
start=utc();tick=time.perf_counter();stages=[]
def record(kind,**kw):
    row=dict(event=kind,utc=utc(),elapsed_seconds=time.perf_counter()-tick,**kw)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def run(name,script,extra,timeout=900):
    cmd=[str(PY),'scripts/'+script,'--output','results/'+name]+extra
    before=utc();t=time.perf_counter();record('stage_start',name=name,command=cmd)
    with (ROOT/'logs'/f'{name}.log').open('x') as f:result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    row=dict(name=name,started_utc=before,finished_utc=utc(),seconds=time.perf_counter()-t,exit_code=result.returncode)
    stages.append(row);record('stage_complete',**row)
    assert result.returncode==0,f'{name} failed; original logs retained'
try:
    record('waiting_for_environment')
    env=ROOT/'results/flashmoba-environment-v0/environment.json'
    while not env.exists():
        assert time.perf_counter()-tick<2700,'Environment did not finish within 45-minute wait budget'
        time.sleep(10)
    assert json.loads(env.read_text())['status']=='complete'
    run('flashmoba-official-gate-v0','verify_flashmoba_official.py',[])
    run('flashmoba-official-prefill-v0','benchmark_flashmoba_official.py',
        ['--gate','results/flashmoba-official-gate-v0/verification.json'])
    run('flashmoba-metadata-gate-v0','verify_flashmoba_metadata.py',
        ['--gate','results/flashmoba-official-gate-v0/verification.json'])
    status='complete';error=None
except Exception as exc:
    status='incomplete';error=dict(message=str(exc),traceback=traceback.format_exc());record('failed',error=error)
(OUT/'batch.json').write_text(json.dumps(dict(status=status,error=error,stages=stages,
    started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,scientific_optimizer_updates=0),indent=2))
if status!='complete':raise SystemExit(1)
