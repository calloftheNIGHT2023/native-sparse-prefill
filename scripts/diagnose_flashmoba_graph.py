"""Isolate the optional graph failure; cached inputs here are component diagnostics, never timing claims."""
import argparse,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def utc():return datetime.now(timezone.utc).isoformat()
def main(a):
    out=ROOT/'results/flashmoba-graph-component-v0'
    if not a.component:
        out.mkdir(parents=True,exist_ok=False);(out/'source.py').write_bytes(Path(__file__).read_bytes());rows=[]
        for component in ['mean','route','sort','attention']:
            start=utc();t=time.perf_counter()
            with (ROOT/'logs'/f'flashmoba-graph-component-{component}-v0.log').open('x') as f:
                r=subprocess.run([sys.executable,str(Path(__file__)), '--component',component],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,timeout=120)
            rows.append(dict(component=component,started_utc=start,finished_utc=utc(),seconds=time.perf_counter()-t,exit_code=r.returncode))
        (out/'index.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True);return
    import torch
    from flash_moba.flash_moba_interface import _wrapped_moba_fused_topk,_wrapped_varlen_sort,flash_moba_attn_varlen_func
    from flash_moba.triton_mean_pool import mean_pool_kernel
    from flashmoba_fixed_metadata import FlashMoBAFixedMetadata
    from benchmark_topk_cuda_graphs import capture
    torch.manual_seed(2026091526);torch.set_num_threads(4);start=utc();t=time.perf_counter()
    try:
        with torch.no_grad():
            n=8192;h=8;d=128;b=128;k=4;m=FlashMoBAFixedMetadata([n],h,d,b,k)
            q,keys,v=[torch.randn(n,h,d,dtype=torch.bfloat16,device='cuda') for _ in range(3)]
            pooled=torch.zeros(n//b,h,d,device='cuda',dtype=q.dtype)
            def mean():
                mean_pool_kernel[(n//b,1,h)](keys,pooled,d,b,m.cu,m.cu_blocks,keys.stride(0),keys.stride(1),pooled.stride(0),pooled.stride(1))
                return pooled
            mean()
            def route():return _wrapped_moba_fused_topk(q,pooled,m.cu,m.cu,m.cu_blocks,n,n,k,b,causal=True)
            offsets,counts,ids=route()
            def sort():return _wrapped_varlen_sort(offsets,counts,ids)
            sorted_ids=sort()
            def attention():return flash_moba_attn_varlen_func(q,keys,v,m.cu,m.cu,n,n,offsets,counts,sorted_ids,m.lg_m,b,dropout_p=0.,causal=True)
            funcs=dict(mean=mean,route=route,sort=sort,attention=attention)
            graph,outputs=capture(funcs[a.component]);graph.replay();torch.cuda.synchronize()
        status='passed';error=None
    except Exception as exc:status='failed';error=dict(message=str(exc),traceback=traceback.format_exc())
    record=dict(status=status,error=error,component=a.component,length=8192,heads=8,dim=128,block=128,topk=4,dtype='bfloat16',
        started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-t,component_test_only=True,scientific_optimizer_updates=0)
    (out/f'{a.component}.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    if status!='passed':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--component',choices=['mean','route','sort','attention']);main(p.parse_args())
