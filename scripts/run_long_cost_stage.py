"""At most fifteen minutes of sequential cost-only work; fail closed and preserve logs."""
import json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/flashmoba-long-cost-controller-v0';OUT.mkdir(parents=True,exist_ok=False)
cfg=json.loads((ROOT/'configs/flashmoba-long-cost-v0.json').read_text());(OUT/'source.py').write_bytes(Path(__file__).read_bytes());(OUT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
start=datetime.now(timezone.utc).isoformat();tick=time.perf_counter();runs=[]
for kind,length in cfg['jobs']:
    remaining=cfg['maximum_stage_seconds']-(time.perf_counter()-tick)
    if remaining<=0:break
    name=f'{kind}-{length}';st=datetime.now(timezone.utc).isoformat();t=time.perf_counter()
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'third_party/flash-moba-barrier-candidate-v0')
    cmd=['/opt/native-sparse-flashmoba-env-v2/bin/python','-u','scripts/run_long_context_cost.py','--precision','amp_bf16','--kind',kind,'--length',str(length)]
    print(json.dumps(dict(event='start',name=name,utc=st)),flush=True)
    with (OUT/(name+'.log')).open('w') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:code=p.wait(timeout=min(remaining,cfg['per_process_timeout_seconds']))
        except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
    row=dict(name=name,kind=kind,length=length,started_utc=st,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-t,returncode=code)
    runs.append(row);(OUT/'partial.json').write_text(json.dumps(runs,indent=2)+'\n');print(json.dumps(row),flush=True)
    if code:
        print((OUT/(name+'.log')).read_text()[-5000:],flush=True);break
result=dict(status='complete' if len(runs)==len(cfg['jobs']) and all(r['returncode']==0 for r in runs) else 'stopped_on_failure_or_limit',
            started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),runs=runs,scientific_optimizer_updates=0)
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
