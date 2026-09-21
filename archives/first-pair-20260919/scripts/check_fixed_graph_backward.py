"""Repeat backward on one saved autograd graph, without rerouting or repooling."""
import functools,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
import torch
import run_flashmoba_realtext_precision as base
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/flashmoba-fixed-graph-backward-v0';out.mkdir(parents=True,exist_ok=False)
start=datetime.now(timezone.utc).isoformat();tick=time.perf_counter()
(out/'source.py').write_bytes(Path(__file__).read_bytes());torch.set_num_threads(4)
p=ROOT/'results/flashmoba-qwen-long-precision-v0/fixed-real-qkv-example.pt'
z=torch.load(p,weights_only=True);q,k,v=[z[n].cuda().requires_grad_() for n in ['q','k','v']]
go=torch.load(ROOT/'results/flashmoba-pool-repair-v1/gradient-repeats.pt',weights_only=True)['upstream_gradient'].cuda()
base.LOCKED_POOL_CONFIG=(32,4,3)
base.flash_moba_attn_varlen_func=functools.partial(base.flash_moba_attn_varlen_func,deterministic=True)
output=base.sparse(q,k,v,128,4,'fp32');grads=[];rows=[]
for repeat in range(4):
    g=torch.autograd.grad(output,(q,k,v),go,retain_graph=True)
    if grads:
        rr=[]
        for a,b in zip(grads[0],g):
            diff=b.float()-a.float();rr.append(dict(relative_l2=float(diff.norm()/a.float().norm()),max_abs=float(diff.abs().max()),changed_elements=int((a!=b).sum())))
        rows.append(dict(repeat=repeat,against=0,qkv=rr));print(json.dumps(rows[-1]),flush=True)
    grads.append(g)
result=dict(status='complete',started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-tick,
    repeats=rows,forward_calls=1,backward_calls=4,deterministic_requested=True,pooling='fp32',input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scientific_optimizer_updates=0,
    interpretation='Same saved forward graph and same upstream gradient. Any variation comes from backward execution, not a new pooling/router call. This alone does not determine a root cause or prove which result is more accurate.')
(out/'result.json').write_text(json.dumps(result,indent=2))
torch.save(dict(gradients=[[t.cpu() for t in g] for g in grads]),out/'gradients.pt')
