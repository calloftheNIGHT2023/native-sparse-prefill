"""Keep numerical routing diagnostics separate from the failed correctness gate."""
import json,sys,time
from pathlib import Path
from datetime import datetime,timezone
import torch
ROOT=Path(__file__).resolve().parents[1]
from verify_flashmoba_official import block_masks,decode_csc
from flash_moba import flash_topk_varlen_func
from flash_moba.triton_mean_pool import flash_topk_mean_pool
import flash_moba_cuda
out=ROOT/'results/flashmoba-routing-diagnostic-v0';out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(2026091525);rows=[]
for dtype in [torch.float16,torch.bfloat16]:
    for lengths,h,hk in [([129,517],4,2),([129,517],4,4),([517],4,2)]:
        n=sum(lengths);b=64;t=4;d=128
        q=torch.randn(n,h,d,device='cuda',dtype=dtype);k=torch.randn(n,hk,d,device='cuda',dtype=dtype)
        cu=torch.tensor([0]+torch.tensor(lengths).cumsum(0).tolist(),device='cuda',dtype=torch.int32)
        km,cm,mx=flash_topk_mean_pool(k,cu,max(lengths),b)
        means=[];st=0
        for le in lengths:means.extend([z.float().mean(0).to(dtype) for z in k[st:st+le].split(b)]);st+=le
        pooled=torch.stack(means)
        offset,count,idx,vals,ids=flash_moba_cuda.moba_fused_topk(q,km,cu,cu,cm,max(lengths),max(lengths),t,b,True)
        masks=decode_csc(offset,count,idx,lengths,h,b);refs,gates=block_masks(q,k,lengths,b,t)
        st=0
        for bi,(le,mask,ref,gate) in enumerate(zip(lengths,masks,refs,gates)):
            pos=torch.arange(le,device='cuda')//b;cur=torch.arange(gate.shape[-1],device='cuda')[None,None,:]==pos[None,:,None]
            finite=gate.masked_fill(cur,0).masked_fill(gate==-torch.inf,0)
            regret=(finite*ref).sum(-1)-(finite*mask).sum(-1)
            worst=regret.argmax();head=int(worst//le);qr=int(worst%le)
            row=dict(dtype=str(dtype),lengths=lengths,heads=h,kv_heads=hk,batch_index=bi,
                mean_max_abs=float((km-pooled).abs().max()),mean_different_elements=int((km!=pooled).sum()),
                changed_rows=int((mask!=ref).any(-1).sum()),max_regret=float(regret.max()),
                worst_head=head,worst_row=qr,reference_gate=gate[head,qr].tolist(),
                raw_kernel_ids=ids[st+qr,head,:t].tolist(),raw_kernel_values=vals[st+qr,head,:t].tolist())
            rows.append(row);print(json.dumps(row),flush=True);st+=le
        torch.save(dict(q=q.cpu(),k=k.cpu(),official_means=km.cpu(),reference_means=pooled.cpu(),
            lengths=lengths,official_topk_ids=ids.cpu(),official_topk_values=vals.cpu()),out/f'{str(dtype)}_b{len(lengths)}_h{h}_hk{hk}.pt')
(out/'diagnostic.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,scientific_optimizer_updates=0),indent=2))
