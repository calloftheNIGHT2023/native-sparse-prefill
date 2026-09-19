"""Independent dense masked reference and numerical/tie audit for fused shortlist."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from block_candidate_topk import select_block_topk
from triton_block_candidate_topk import select_fused_block_topk
from triton_selected_attention import triton_selected_attention


def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);start=utc();tick=time.perf_counter();rows=[]
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(2026091514)
    sources={p:sha(ROOT/p) for p in ['src/triton_block_candidate_topk.py','src/block_candidate_topk.py',
        'src/triton_selected_attention.py','scripts/verify_fused_block_candidate.py']}
    for p in sources:
        dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    try:
        for dtype in [torch.float32,torch.bfloat16,torch.float16]:
            for n,d,block,routes,offset in [(8,16,8,1,0),(37,16,8,2,3),(129,32,16,4,7),(257,128,32,4,16),(1024,128,32,4,0)]:
                for method in ['mean','minmax']:
                    storage=torch.randn(2,n,3,d,device='cuda',dtype=dtype)
                    q,k,v=[t.detach().requires_grad_() for t in storage.unbind(2)]
                    ids=select_fused_block_topk(q,k,block,routes,method,offset)
                    ref=select_block_topk(q,k,block,routes,method,32,offset)
                    valid=ids>=0;pos=torch.arange(n,device='cuda')
                    assert bool(((ids<0)|(ids<=pos[None,:,None])).all())
                    assert bool((ids[:,:,0]==pos[None]).all())
                    assert bool((ids[:,:,1]==(pos-1)[None]).all())
                    ordered=ids.sort(-1).values
                    assert not bool(((ordered[:,:,1:]==ordered[:,:,:-1])&(ordered[:,:,1:]>=0)).any())
                    same=(ids.sort(-1).values==ref.sort(-1).values).all(-1)
                    # If a boundary tie changes support, compare selected score order independently.
                    full=q.detach().double()@k.detach().double().transpose(1,2)/(d**.5)
                    selected=full.gather(-1,ids.clamp_min(0)).masked_fill(~valid,-torch.inf).sort(-1).values
                    reference=full.gather(-1,ref.clamp_min(0)).masked_fill(ref<0,-torch.inf).sort(-1).values
                    torch.testing.assert_close(selected,reference,atol=2e-5,rtol=2e-5)
                    # Gradients compared to independent ordinary dense-mask autograd on SAME support.
                    mask=torch.zeros(2,n,n,dtype=torch.bool,device='cuda')
                    for s in range(8):
                        gs,qs=torch.where(ids[:,:,s]>=0);mask[gs,qs,ids[gs,qs,s]]=True
                    qr,kr,vr=[t.detach().float().requires_grad_() for t in [q,k,v]]
                    expected=(qr@(kr/(d**.5)).transpose(1,2)).masked_fill(~mask,-torch.inf).softmax(-1)@vr
                    actual=triton_selected_attention(q,k,v,ids)
                    go=torch.randn_like(actual)
                    ga=torch.autograd.grad(actual,[q,k,v],go)
                    gr=torch.autograd.grad(expected,[qr,kr,vr],go.float())
                    atol=0.025 if dtype==torch.bfloat16 else 0.004 if dtype==torch.float16 else 2e-4
                    torch.testing.assert_close(actual.float(),expected,atol=atol,rtol=atol)
                    for x,y in zip(ga,gr):torch.testing.assert_close(x.float(),y,atol=atol,rtol=atol)
                    row=dict(dtype=str(dtype),length=n,dim=d,block=block,routes=routes,offset=offset,method=method,
                        support_changed_rows=int((~same).sum()),output_max_abs=float((actual.float()-expected).abs().max()),
                        gradient_max_abs=max(float((x.float()-y).abs().max()) for x,y in zip(ga,gr)),status='passed')
                    rows.append(row);print(json.dumps(row),flush=True)
        status='passed';error=None
    except Exception as exc:status='failed';error=dict(message=str(exc),traceback=traceback.format_exc());print(error,flush=True)
    result=dict(status=status,error=error,started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,
        checks=rows,sources=sources,gpu=torch.cuda.get_device_name(),torch=torch.__version__,scientific_optimizer_updates=0,
        protected_files_unchanged=all(sha(ROOT/p)==h for p,h in sources.items()))
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if status!='passed':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
