"""Controlled reduction-config intervention on saved real QKV; source unchanged."""
import json,hashlib,time,subprocess,traceback
from pathlib import Path
from datetime import datetime,timezone
import torch
from flash_moba.triton_mean_pool import mean_pool_kernel
from flash_moba.flash_moba_interface import flash_moba_attn_varlen_func,decide_lg_block_m
import flash_moba_cuda
ROOT=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=ROOT/'results/flashmoba-pool-autotune-v0';out.mkdir(parents=True,exist_ok=False)
start=utc();tick=time.perf_counter();(out/'source.py').write_bytes(Path(__file__).read_bytes())
path=ROOT/'results/flashmoba-qwen-long-precision-v0/fixed-real-qkv-example.pt'
z=torch.load(path,weights_only=True);q,k,v=[z[x].cuda() for x in ['q','k','v']]
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
n,h,d=q.shape;hk=k.shape[1];b=128;nb=(n+b-1)//b
cu=torch.tensor([0,n],device='cuda',dtype=torch.int32);cm=torch.tensor([0,nb],device='cuda',dtype=torch.int32)
configs=[(32,2,3),(32,4,3),(32,4,4),(64,2,3),(64,4,3),(64,4,4),(64,8,3),(128,2,3),(128,4,3),(128,4,4),(128,8,3),(128,8,4)]
ideal=torch.stack([x.double().mean(0) for x in k.split(b)]).float()
def pool(dtype,c):
    inp=k.to(dtype);means=torch.zeros((nb,hk,d),device='cuda',dtype=dtype)
    bn,warps,stages=c
    mean_pool_kernel.fn[(nb,1,hk)](inp,means,d,b,cu,cm,inp.stride(0),inp.stride(1),means.stride(0),means.stride(1),kBlockN=bn,num_warps=warps,num_stages=stages)
    return means.to(k.dtype)
def route(means,topk):
    offsets,counts,idx,_,ids=flash_moba_cuda.moba_fused_topk(q,means,cu,cu,cm,n,n,topk,b,True)
    idx=flash_moba_cuda.varlen_sort(offsets.flatten(),(offsets+counts).flatten(),idx)
    result=flash_moba_attn_varlen_func(q,k,v,cu,cu,n,n,offsets,counts,idx,decide_lg_block_m(topk,b,n,True),b,dropout_p=0.,causal=True)
    return ids[...,:topk].sort(-1).values,result
rows=[];means_saved={};outputs={}
try:
    with torch.inference_mode():
        for dtype in [torch.bfloat16,torch.float32]:
            for c in configs:
                key=f'{str(dtype)}_bn{c[0]}_w{c[1]}_s{c[2]}'
                a=pool(dtype,c);a2=pool(dtype,c);assert torch.equal(a,a2)
                means_saved[key]=a.cpu()
                for topk in [2,4]:
                    ids,output=route(a,topk);ids2,output2=route(a2,topk)
                    assert torch.equal(ids,ids2) and torch.equal(output,output2)
                    basekey=(str(dtype),topk)
                    if basekey not in outputs:outputs[basekey]=(a.clone(),ids.clone(),output.clone())
                    ma,ia,oa=outputs[basekey]
                    row=dict(dtype=str(dtype),kBlockN=c[0],num_warps=c[1],num_stages=c[2],topk=topk,
                        pooling_elements_changed_from_first=int((a!=ma).sum()),
                        pooling_max_error_vs_fp64_mean=float((a.float()-ideal).abs().max()),
                        route_rows_changed_from_first=int((ids!=ia).any(-1).sum()),total_route_rows=n*h,
                        output_relative_difference=float((output.float()-oa.float()).norm()/oa.float().norm()),
                        repeat_bitwise_equal=True)
                    rows.append(row);print(json.dumps(row),flush=True)
    torch.save(means_saved,out/'pooled-means-by-config.pt')
    result=dict(status='complete',started_utc=start,finished_utc=utc(),seconds=time.perf_counter()-tick,
                input_sha256=sha(path),input_file=str(path.relative_to(ROOT)),source_sha256=sha(Path(__file__)),conditions=rows,
                fixed_input_shape=list(q.shape),kv_shape=list(k.shape),scientific_optimizer_updates=0,
                earlier_process_autotune_choices_not_recorded=True,not_a_fullmodel_quality_evaluation=True)
except Exception:
    result=dict(status='failed',started_utc=start,finished_utc=utc(),error=traceback.format_exc(),conditions=rows)
    raise
finally:(out/'diagnostic.json').write_text(json.dumps(result,indent=2))
