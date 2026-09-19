"""Small CUDA correctness gate, without optimizers or cloud operations."""
import argparse,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from chunked_topk_attention import ChunkedTopKAttention,select_causal_topk,selected_attention
from zoology_sparse_schedule import ScheduledAttention

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(a):
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats();started=now();tick=time.perf_counter();checks=[]
    sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/chunked_topk_attention.py',ROOT/'src/zoology_sparse_schedule.py']}
    for dtype in [torch.float64,torch.float32]:
        torch.manual_seed(2026091511)
        x=torch.randn(2,37,3,2,16,device='cuda',dtype=dtype,requires_grad=True)
        y=x.detach().clone().requires_grad_(True);go=torch.randn(2,37,2,16,device='cuda',dtype=dtype)
        old=ScheduledAttention(0,'native',dropout=0).eval();new=ChunkedTopKAttention(dropout=0,query_chunk=11).eval()
        u,w=old(x),new(y)
        da,=torch.autograd.grad(u,x,go);db,=torch.autograd.grad(w,y,go)
        tol=1e-10 if dtype==torch.float64 else 5e-5
        torch.testing.assert_close(u,w,rtol=tol,atol=tol);torch.testing.assert_close(da,db,rtol=tol*10,atol=tol)
        checks.append(dict(dtype=str(dtype),reference='legacy full-matrix top-k',
            output_max_abs_error=float((u-w).abs().max()),gradient_max_abs_error=float((da-db).abs().max())))
    for dtype in [torch.float16,torch.bfloat16]:
        if dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():
            checks.append(dict(dtype=str(dtype),status='unsupported'));continue
        torch.manual_seed(2026091512)
        q,k,v=[torch.randn(2,29,16,device='cuda',dtype=dtype,requires_grad=True) for _ in range(3)]
        ids=select_causal_topk(q,k,8,2,7);keep=torch.rand(ids.shape,device='cuda')>=.2
        actual=selected_attention(q,k,v,ids,.2,7,keep)
        mask=torch.zeros(2,29,29,device='cuda',dtype=torch.bool)
        drop=torch.zeros(2,29,29,device='cuda',dtype=torch.float32)
        for g in range(2):
            for i in range(29):
                ok=ids[g,i]>=0;mask[g,i,ids[g,i,ok]]=True;drop[g,i,ids[g,i,ok]]=keep[g,i,ok].float()/.8
        scores=q.float()@(k.float()/4).transpose(1,2)
        expected=((scores.masked_fill(~mask,-torch.inf).softmax(-1)*drop)@v.float()).to(dtype)
        go=torch.randn_like(actual);da=torch.autograd.grad(actual,(q,k,v),go,retain_graph=True)
        db=torch.autograd.grad(expected,(q,k,v),go)
        torch.testing.assert_close(actual,expected,rtol=.02,atol=.02)
        for a1,b1 in zip(da,db):torch.testing.assert_close(a1,b1,rtol=.03,atol=.03)
        checks.append(dict(dtype=str(dtype),reference='independent dense autograd with identical support, FP32 accumulation and fixed dropout',
            output_max_abs_error=float((actual-expected).abs().max()),
            gradient_max_abs_error=max(float((a1-b1).abs().max()) for a1,b1 in zip(da,db)),
            limitation='Not equivalence to legacy half-precision ranking or checkpoint accuracy'))
    torch.cuda.synchronize()
    result=dict(status='passed',started_utc=started,finished_utc=now(),wall_seconds=time.perf_counter()-tick,
                torch_version=torch.__version__,python=sys.version,cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),checks=checks,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                optimizer_updates=0,new_cloud_resources=0,sources=sources)
    (out/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);main(p.parse_args())
