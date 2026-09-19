"""Selected-score optimality audit; tolerate only measured GEMM rounding errors."""
import argparse,hashlib,json,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=datetime.now(timezone.utc).isoformat();tick=time.perf_counter();checks=[]
    files=['src/triton_streaming_selector.py','scripts/verify_streaming_topk_selector.py']
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    for p in files:
        dst=out/'source'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/p).read_bytes())
    try:
        from triton_streaming_selector import select_streaming_topk
        from triton_topk_selector import select_triton_topk
        torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(2026091561)
        for dtype in [torch.bfloat16,torch.float16]:
            for n,d in [(8,16),(37,32),(129,128),(1024,128)]:
                q,k=[torch.randn(2,n,d,device='cuda',dtype=dtype) for _ in range(2)]
                ids=select_streaming_topk(q,k);ref=select_triton_topk(q,k,8,2,256)
                pos=torch.arange(n,device='cuda');sort=ids.sort(-1).values
                assert bool(((ids<0)|(ids<=pos[None,:,None])).all())
                assert bool(((sort[:,:,1:]!=sort[:,:,:-1])|(sort[:,:,1:]<0)).all())
                assert bool(((ids>=0).sum(-1)==torch.minimum(pos+1,torch.tensor(8,device='cuda'))[None,:]).all())
                assert torch.equal(ids[:,:,0],pos.expand(2,-1))
                assert torch.equal(ids[:,:,1],(pos-1).expand(2,-1))
                # Compare ranked value lists on a shared CPU double-precision score matrix.
                # This checks mathematical top-k optimality separately from BF16/FP16 tie identity.
                scale=(k/d**.5).double();score=torch.bmm(q.double(),scale.transpose(1,2))
                allowed=pos[None,:]<=pos[:,None]-2;score=score.masked_fill(~allowed,-torch.inf)
                wanted=score.topk(6,dim=-1).values
                got=score.gather(-1,ids[:,:,2:].clamp_min(0)).masked_fill(ids[:,:,2:]<0,-torch.inf).sort(-1,descending=True).values
                finite=torch.isfinite(wanted);max_gap=float((wanted[finite]-got[finite]).abs().max())
                # Each rounded score has <= one ULP error; support is explicitly allowed to change at near ties.
                eps=torch.finfo(dtype).eps;bound=2*eps*torch.maximum(wanted[finite].abs(),torch.ones_like(wanted[finite]))
                assert bool(((wanted[finite]-got[finite]).abs()<=bound+1e-5).all()), 'Selected scores not near-optimal'
                changed=int((ids.sort(-1).values!=ref.sort(-1).values).any(-1).sum())
                checks.append(dict(dtype=str(dtype),length=n,dim=d,support_rows_different=changed,
                    max_selected_score_gap_fp64_reference=max_gap,status='passed'))
                print(json.dumps(checks[-1]),flush=True)
        status='passed';error=None
    except Exception as exc:
        status='failed';error=dict(message=str(exc),traceback=traceback.format_exc())
    result=dict(status=status,error=error,started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-tick,checks=checks,sources=sources,scientific_optimizer_updates=0,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        limitation='Near-optimal selected scores only; not exact support identity or model-quality equivalence')
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if status!='passed':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
