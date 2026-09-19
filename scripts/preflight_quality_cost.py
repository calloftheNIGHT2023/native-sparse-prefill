"""Additional K16 forward/backward causal checks for the planned pilot."""
from pathlib import Path
import json,time,traceback,hashlib
from datetime import datetime,timezone
import torch
import run_flashmoba_realtext_precision as base
from verify_flashmoba_official_v1 import block_masks,reference
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-quality-cost-k16-gate-v0';OUT.mkdir(parents=True,exist_ok=False)
start=datetime.now(timezone.utc).isoformat();tick=time.perf_counter();rows=[]
(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
try:
    torch.set_num_threads(4);torch.manual_seed(2026091551);torch.backends.cuda.matmul.allow_tf32=False
    assert hashlib.sha256(Path(base.flash_moba_cuda.__file__).read_bytes()).hexdigest()=='72c3fda9e7d4da701bfc76b0e32f4d80fdbeec70d462c83b87b97f51d887ed1c'
    base.LOCKED_POOL_CONFIG=(32,4,3)
    for n in [2048,4096]:
        q=torch.randn(n,14,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn(n,2,64,device='cuda',dtype=torch.bfloat16,requires_grad=True);v=torch.randn_like(k,requires_grad=True)
        y,_,means=base.sparse(q,k,v,128,16,'fp32',details=True)
        masks,_=block_masks(q,k,[n],128,16,provided_means=means)
        qr,kr,vr=[x.detach().float().requires_grad_() for x in [q,k,v]]
        ref=reference(qr,kr,vr,[n],masks,128);go=torch.randn_like(y)
        ga=torch.autograd.grad(y,(q,k,v),go);gr=torch.autograd.grad(ref,(qr,kr,vr),go.float())
        torch.testing.assert_close(y.float(),ref,rtol=.03,atol=.03)
        for a,b in zip(ga,gr):torch.testing.assert_close(a.float(),b,rtol=.03,atol=.03)
        with torch.no_grad():
            kk=k.clone();vv=v.clone();kk[n//2:]=torch.randn_like(kk[n//2:])*5;vv[n//2:]=torch.randn_like(vv[n//2:])*5
            changed=base.sparse(q,kk,vv,128,16,'fp32')
            assert torch.equal(y[:n//2],changed[:n//2])
        row=dict(length=n,topk=16,output_max_abs=float((y-ref).abs().max()),gradient_max_abs=max(float((a-b).abs().max()) for a,b in zip(ga,gr)),causal_prefix_unchanged=True)
        rows.append(row);print(json.dumps(row),flush=True)
        del q,k,v,y,means,masks,qr,kr,vr,ref,go,ga,gr,kk,vv,changed
    result=dict(status='passed',rows=rows)
except Exception:result=dict(status='failed',rows=rows,error=traceback.format_exc())
result.update(started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-tick,scientific_optimizer_updates=0)
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if result['status']!='passed':raise SystemExit(1)
