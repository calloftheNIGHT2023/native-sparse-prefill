"""Check metadata-only wrapper equals original API, including changed inputs after graph capture."""
import argparse,hashlib,json,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flashmoba_fixed_metadata import FlashMoBAFixedMetadata
from benchmark_topk_cuda_graphs import capture
from flash_moba import flash_moba_varlen_func
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();start=utc();checks=[]
    sources={p:sha(ROOT/p) for p in ['scripts/verify_flashmoba_metadata.py','src/flashmoba_fixed_metadata.py','scripts/benchmark_topk_cuda_graphs.py']}
    for p in sources:
        dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    try:
        gate=json.loads(a.gate.read_text());assert gate['status']=='passed' and len(gate['checks'])==12
        torch.manual_seed(2026091522);torch.set_num_threads(4)
        for dtype in [torch.float16,torch.bfloat16]:
            for lengths,h,hk,d,b,t in [([257],1,1,64,64,2),([129,517],4,2,128,64,4),([8192],8,8,128,128,4)]:
                model=FlashMoBAFixedMetadata(lengths,h,d,b,t,hk)
                q=torch.randn(sum(lengths),h,d,device='cuda',dtype=dtype,requires_grad=True)
                k=torch.randn(sum(lengths),hk,d,device='cuda',dtype=dtype,requires_grad=True);v=torch.randn_like(k,requires_grad=True)
                official=lambda:flash_moba_varlen_func(q,k,v,model.cu,model.cu,max(lengths),max(lengths),b,t,causal=True)
                actual=model(q,k,v);expected=official();torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                go=torch.randn_like(actual);ga=torch.autograd.grad(actual,(q,k,v),go);gr=torch.autograd.grad(expected,(q,k,v),go)
                tol=.025 if dtype==torch.bfloat16 else .004
                for x,y in zip(ga,gr):torch.testing.assert_close(x,y,atol=tol,rtol=tol)
                with torch.no_grad():
                    fn=lambda:model(q,k,v);graph,output=capture(fn)
                    before=output.clone()
                    for z in (q,k,v):z.copy_(torch.randn_like(z))
                    new_expected=official();graph.replay();torch.cuda.synchronize()
                    torch.testing.assert_close(output,new_expected,atol=0,rtol=0)
                    assert not torch.equal(before,output)
                row=dict(dtype=str(dtype),lengths=lengths,heads=h,kv_heads=hk,dim=d,block=b,topk=t,
                    eager_exact_equal=True,gradient_close=True,changed_input_graph_exact_equal=True,status='passed')
                checks.append(row);print(json.dumps(dict(utc=utc(),**row)),flush=True)
                del graph,output,actual,expected,ga,gr,model,q,k,v
        status='passed';error=None
    except Exception as exc:status='failed';error=dict(message=str(exc),traceback=traceback.format_exc());print(error,flush=True)
    (out/'verification.json').write_text(json.dumps(dict(status=status,error=error,checks=checks,sources=sources,
        started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,scientific_optimizer_updates=0),indent=2))
    if status!='passed':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);main(p.parse_args())
