"""Three fixed reduction sizes on identical saved8K contexts; finite ordered batch."""
import json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/flashmoba-fixed-pool-batch-v0';out.mkdir(parents=True,exist_ok=False)
started=datetime.now(timezone.utc).isoformat();rows=[];tick=time.perf_counter()
base=json.loads((ROOT/'data/flashmoba-qwen-long-precision-v0/config.json').read_text())
for bn in [32,64,128]:
    c=dict(base,pool_config=[bn,4,3],probe_contexts=0,controlled_reduction_intervention=True,
           scope='Same16 saved8K Qwen0.5B contexts and unchanged weights; fixed reduction kBlockN,4warps,3stages. FP32 backbone/BF16 attention. BF16 versus FP32 pooled accumulation. No new independent data or training; no global novelty claim.')
    cp=ROOT/f'configs/flashmoba-fixed-pool-bn{bn}-v0.json';assert not cp.exists();cp.write_text(json.dumps(c,indent=2))
    name=f'flashmoba-fixed-pool-bn{bn}-v0';st=datetime.now(timezone.utc).isoformat();t=time.perf_counter()
    with (ROOT/'logs'/f'{name}.log').open('w') as log:
        p=subprocess.run([sys.executable,str(ROOT/'scripts/run_flashmoba_realtext_precision.py'),'--data',str(ROOT/'data/flashmoba-qwen-long-precision-v0'),'--config',str(cp),'--output',str(ROOT/'results'/name)],stdout=log,stderr=subprocess.STDOUT)
    rows.append(dict(name=name,started_utc=st,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-t,exit_code=p.returncode))
    (out/'batch.json').write_text(json.dumps(dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-tick,status='running' if p.returncode==0 else 'failed',runs=rows),indent=2))
    print(json.dumps(rows[-1]),flush=True)
    if p.returncode:raise RuntimeError(name)
(out/'batch.json').write_text(json.dumps(dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-tick,status='complete',runs=rows),indent=2))
