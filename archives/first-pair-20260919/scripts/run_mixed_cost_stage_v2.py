"""Sequential bounded controller; preserve configuration failures without adding runs."""
import os,json,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-mixed-cost-controller-v2';OUT.mkdir(parents=True,exist_ok=False)
cfg=json.loads((ROOT/'configs/flashmoba-mixed-cost-v2.json').read_text())
(OUT/'source.py').write_bytes(Path(__file__).read_bytes());(OUT/'frozen-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
start=datetime.now(timezone.utc).isoformat();rows=[]
for precision,length in cfg['configurations']:
    name=f'{precision}-{length}';s=datetime.now(timezone.utc).isoformat();t=time.perf_counter()
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'third_party/flash-moba-barrier-candidate-v0')
    cmd=['/opt/native-sparse-flashmoba-env-v2/bin/python','-u','scripts/run_mixed_precision_cost_v2.py','--precision',precision,'--length',str(length)]
    print(json.dumps(dict(event='start',name=name,utc=s)),flush=True)
    with (OUT/(name+'.log')).open('w') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:code=p.wait(timeout=cfg['per_process_timeout_seconds'])
        except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
    row=dict(name=name,started_utc=s,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-t,returncode=code)
    rows.append(row);print(json.dumps(row),flush=True)
    if code:print((OUT/(name+'.log')).read_text()[-4000:],flush=True)
    (OUT/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
    # A failed small precision gate must not be followed by the larger case of that precision.
    if code:break
result=dict(status='complete' if len(rows)==1 and all(x['returncode']==0 for x in rows) else 'stopped_on_failure',
            started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),runs=rows,scientific_optimizer_updates=0)
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
