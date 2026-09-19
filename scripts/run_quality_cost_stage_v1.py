"""Sequential fixed-budget controller: ten independent processes on one GPU."""
import json,os,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-quality-cost-controller-v1';OUT.mkdir(parents=True,exist_ok=False)
PY=Path('/opt/native-sparse-flashmoba-env-v2/bin/python')
DATA=ROOT/'data/flashmoba-quality-cost-v0'
CAND=ROOT/'third_party/flash-moba-barrier-candidate-v0'
cfg=json.loads((DATA/'config.json').read_text())
assert json.loads((ROOT/'results/flashmoba-quality-cost-k16-gate-v0/result.json').read_text())['status']=='passed'
start=datetime.now(timezone.utc).isoformat();tick=time.perf_counter();completed=[]
(OUT/'source.py').write_bytes(Path(__file__).read_bytes());(OUT/'frozen-config.json').write_bytes((DATA/'config.json').read_bytes())
def event(kind,**kw):
    row=dict(utc=datetime.now(timezone.utc).isoformat(),event=kind,**kw)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
try:
    for si,seed in enumerate(cfg['initialization_seeds']):
        order=cfg['conditions'] if si==0 else list(reversed(cfg['conditions']))
        for cond in order:
            name=f"{cond['name']}-s{seed}";folder=ROOT/'results'/f'flashmoba-quality-cost-{name}-v1'
            env=os.environ.copy();env.pop('PYTHONPATH',None)
            if cond['extension']=='barrier':env['PYTHONPATH']=str(CAND)
            cmd=[str(PY),'-u','scripts/run_quality_cost_pilot.py','--data',str(DATA),'--output',str(folder),'--condition',cond['name'],'--seed',str(seed)]
            st=time.perf_counter();event('run_start',name=name,command=cmd)
            with (OUT/f'{name}.log').open('w') as f:
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                event('process_started',name=name,pid=p.pid)
                try:code=p.wait(timeout=600)
                except subprocess.TimeoutExpired:p.kill();p.wait();raise
            event('run_end',name=name,returncode=code,seconds=time.perf_counter()-st)
            if code:raise RuntimeError(f'{name} failed: '+(OUT/f'{name}.log').read_text()[-2000:])
            x=json.loads((folder/'result.json').read_text());assert x['status']=='complete' and x['scientific_optimizer_updates']==64
            completed.append(dict(name=name,result_directory=str(folder.relative_to(ROOT)),updates=64))
            (OUT/'partial.json').write_text(json.dumps(completed,indent=2)+'\n')
    result=dict(status='complete',started_utc=start,completed=completed,scientific_optimizer_updates=sum(x['updates'] for x in completed))
    event('stage_complete',trajectories=len(completed),optimizer_updates=result['scientific_optimizer_updates'])
except Exception:
    result=dict(status='failed',started_utc=start,completed=completed,error=traceback.format_exc());event('stage_failed',error=result['error'])
result.update(finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-tick)
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if result['status']!='complete':raise SystemExit(1)
