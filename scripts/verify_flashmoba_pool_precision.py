"""On saved diagnostic inputs, check FP32 pooling and route scoring separately."""
import hashlib,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
import torch
from flash_moba.triton_mean_pool import flash_topk_mean_pool
from verify_flashmoba_official_v1 import block_masks,decode_csc
import flash_moba_cuda
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/flashmoba-pool-precision-v0';out.mkdir(parents=True,exist_ok=False)
tick=time.perf_counter();start=datetime.now(timezone.utc).isoformat();rows=[];torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
for path in sorted((ROOT/'results/flashmoba-routing-diagnostic-v0').glob('*.pt')):
    z=torch.load(path,weights_only=True);q=z['q'].cuda();k=z['k'].cuda();lengths=z['lengths'];b=64;t=4
    cu=torch.tensor([0]+torch.tensor(lengths).cumsum(0).tolist(),device='cuda',dtype=torch.int32)
    original,cm,_=flash_topk_mean_pool(k,cu,max(lengths),b)
    f32,_,_=flash_topk_mean_pool(k.float(),cu,max(lengths),b)
    ideal=[];st=0
    for n in lengths:ideal.extend([p.float().mean(0) for p in k[st:st+n].split(b)]);st+=n
    ideal=torch.stack(ideal);torch.testing.assert_close(f32,ideal,atol=5e-7,rtol=2e-6)
    masksets=[];regrets=[]
    for pooled in [original,f32.to(k.dtype)]:
        offsets,counts,idx,_,_=flash_moba_cuda.moba_fused_topk(q,pooled,cu,cu,cm,max(lengths),max(lengths),t,b,True)
        masks=decode_csc(offsets,counts,idx,lengths,q.shape[1],b)
        refs,scores=block_masks(q,k,lengths,b,t,provided_means=pooled)
        worst=0.
        for mask,ref,gate,n in zip(masks,refs,scores,lengths):
            pos=torch.arange(n,device='cuda')//b;cur=torch.arange(gate.shape[-1],device='cuda')[None,None,:]==pos[None,:,None]
            finite=gate.masked_fill(cur,0).masked_fill(gate==-torch.inf,0)
            worst=max(worst,float(((finite*ref).sum(-1)-(finite*mask).sum(-1)).max()))
        assert worst<2e-4;regrets.append(worst);masksets.append(masks)
    row=dict(input_file=path.name,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),dtype=str(k.dtype),lengths=lengths,
        heads=q.shape[1],kv_heads=k.shape[1],official_pool_max_abs_vs_fp32=float((original.float()-ideal).abs().max()),
        fp32_pool_max_abs=float((f32-ideal).abs().max()),
        changed_query_head_rows=sum(int((a!=b).any(-1).sum()) for a,b in zip(*masksets)),
        total_query_head_rows=sum(lengths)*q.shape[1],conditional_route_max_regrets=regrets)
    rows.append(row);print(json.dumps(row),flush=True)
(out/'verification.json').write_text(json.dumps(dict(status='passed',checks=rows,started_utc=start,
    finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-tick,scientific_optimizer_updates=0,
    interpretation='Numerical pooling deviation; not evidence of downstream quality improvement or original algorithm'),indent=2))
