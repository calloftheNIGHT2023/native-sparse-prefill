"""Verify ordinary in-kernel FP32 repair and microbenchmark its cost on real QKV."""
import sys,json,hashlib,time,functools
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime,timezone
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_flashmoba_realtext_precision as base
from src.flashmoba_fp32_pool import fp32_mean_pool_kernel
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=ROOT/'results/flashmoba-pool-repair-v1';out.mkdir(parents=True,exist_ok=False);start=utc();tick=time.perf_counter()
(out/'source.py').write_bytes(Path(__file__).read_bytes());(out/'kernel.py').write_bytes((ROOT/'src/flashmoba_fp32_pool.py').read_bytes())
torch.set_num_threads(4);torch.manual_seed(2026091546);torch.backends.cuda.matmul.allow_tf32=False
p=ROOT/'results/flashmoba-qwen-long-precision-v0/fixed-real-qkv-example.pt';z=torch.load(p,weights_only=True)
q,k,v=[z[x].cuda() for x in ['q','k','v']];original=base.mean_pool_kernel
base.flash_moba_attn_varlen_func=functools.partial(base.flash_moba_attn_varlen_func,deterministic=True)
configs=[(32,2,3),(32,4,3),(32,4,4),(64,2,3),(64,4,3),(64,4,4),(64,8,3),(128,2,3),(128,4,3),(128,4,4),(128,8,3),(128,8,4)]
def run(mode,c,details=False):
    base.mean_pool_kernel=SimpleNamespace(fn=fp32_mean_pool_kernel) if mode=='in_kernel_fp32' else original
    base.LOCKED_POOL_CONFIG=c
    return base.sparse(q,k,v,128,4,'fp32' if mode=='external_fp32' else 'official',details=details)
rows=[]
with torch.no_grad():
    ref,ri,rm=run('external_fp32',(32,4,3),True)
    for c in configs:
        val,ids,means=run('in_kernel_fp32',c,True)
        assert torch.equal(means,rm) and torch.equal(ids,ri) and torch.equal(val,ref)
        rows.append(dict(config=c,pooled_bitwise_equal=True,route_bitwise_equal=True,output_bitwise_equal=True))
    # CUDA-event timing after warm-up, five interleaved rounds; no kernel-only -> full model inference.
    times=[]
    for repeat in range(5):
        for mode in ['original','external_fp32','in_kernel_fp32'][repeat%3:]+['original','external_fp32','in_kernel_fp32'][:repeat%3]:
            for _ in range(20):run(mode,(32,4,3))
            a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)];a.record()
            for _ in range(100):run(mode,(32,4,3))
            b.record();b.synchronize();times.append(dict(round=repeat,mode=mode,attention_forward_ms=a.elapsed_time(b)/100))
            print(json.dumps(times[-1]),flush=True)
q.requires_grad_(True);k.requires_grad_(True);v.requires_grad_(True);go=torch.randn_like(q)
gradients={};outputs={}
for name,mode in [('external_a','external_fp32'),('external_b','external_fp32'),('fix_a','in_kernel_fp32'),('fix_b','in_kernel_fp32')]:
    val=run(mode,(32,4,3));outputs[name]=val.detach().clone()
    gradients[name]=torch.autograd.grad(val,(q,k,v),go)
torch.save(dict(gradients={k:[x.cpu() for x in v] for k,v in gradients.items()},upstream_gradient=go.cpu()),out/'gradient-repeats.pt')
gradient_checks=[]
for left,right in [('external_a','external_b'),('fix_a','fix_b'),('external_a','fix_a'),('external_b','fix_b')]:
    items=[]
    for a,b in zip(gradients[left],gradients[right]):
        aa=a.float();bb=b.float();delta=aa-bb
        items.append(dict(relative_l2=float(delta.norm()/aa.norm()),max_abs=float(delta.abs().max()),changed_elements=int((a!=b).sum()),elements=a.numel()))
    gradient_checks.append(dict(left=left,right=right,forward_bitwise_equal=torch.equal(outputs[left],outputs[right]),qkv=items))
    print(json.dumps(gradient_checks[-1]),flush=True)
result=dict(status='complete',started_utc=start,finished_utc=utc(),seconds=time.perf_counter()-tick,input_sha256=sha(p),
    q_shape=list(q.shape),kv_shape=list(k.shape),topk=4,block=128,gpu=torch.cuda.get_device_name(),conditions=rows,timing=times,
    gradients_bitwise_equal_to_external_fp32=all(torch.equal(x,y) for x,y in zip(gradients['external_a'],gradients['fix_a'])),
    gradient_checks=gradient_checks,scientific_optimizer_updates=0,previous_v0_failed_bitwise_gradient_assertion=True,
    scope='12 configurations, one stored real layer0 QKV, K4. Timing covers sparse attention routing+forward, not whole model/training. Ordinary precision repair, no novelty claim.')
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status='complete',seconds=result['seconds'])),flush=True)
