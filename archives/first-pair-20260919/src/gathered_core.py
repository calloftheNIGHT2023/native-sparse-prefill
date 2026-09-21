"""Autograd reference that computes main QK only for selected blocks and tail.

This materializes per-query K/V gathers, so it is NOT an efficient GPU kernel.
The indexer and layout may still have quadratic storage/work. No speed claim.
"""
import torch


def gather_plan(selected,layout):
    n,blocks=selected.shape; device=selected.device
    if blocks==0:
        block_ids=torch.empty(n,0,dtype=torch.long,device=device)
        block_valid=torch.empty(n,0,dtype=torch.bool,device=device)
        token_ids=block_ids; token_valid=block_valid
    else:
        count=int(selected.sum(-1).max())
        block_ids=selected.to(torch.int32).topk(count,dim=-1).indices
        block_valid=selected.gather(1,block_ids)
        token_ids=layout.block_tokens.to(device)[block_ids].flatten(1)
        token_valid=block_valid.repeat_interleave(layout.block_tokens.shape[1],dim=-1)
    r=layout.block_tokens.shape[1]
    tail_length=(layout.positions.to(device)+1)%r
    offsets=torch.arange(r,device=device)
    tail_ids=torch.arange(n,device=device)[:,None]-tail_length[:,None]+1+offsets
    tail_valid=offsets[None]<tail_length[:,None]
    tail_ids=tail_ids.clamp(0,n-1)
    ids=torch.cat([token_ids,tail_ids],dim=-1)
    valid=torch.cat([token_valid,tail_valid],dim=-1)
    if not bool(valid.any(-1).all()): raise ValueError('Every query needs visible support')
    return ids,valid,block_ids,block_valid


def gathered_attention(q,k,v,selected,layout,softmax_fp32=True):
    """Q/K/V [heads,tokens,dim]; return output and detached block teacher."""
    ids,valid,block_ids,block_valid=gather_plan(selected,layout)
    keys=k[:,ids,:]; values=v[:,ids,:]
    logits=torch.einsum('hnd,hnmd->hnm',q,keys)*q.shape[-1]**-.5
    logits=logits.masked_fill(~valid[None],-torch.inf)
    probabilities=logits.softmax(-1,dtype=torch.float32 if softmax_fp32 else logits.dtype).to(v.dtype)
    output=torch.einsum('hnm,hnmd->hnd',probabilities,values)
    with torch.no_grad():
        n,b=selected.shape; r=layout.block_tokens.shape[1]; count=block_ids.shape[-1]
        target=probabilities.new_zeros(n,b)
        if count:
            pooled=probabilities.mean(0)[:,:count*r].reshape(n,count,r).amax(-1)*block_valid
            target.scatter_(1,block_ids,pooled)
            target=target/target.sum(-1,keepdim=True).clamp_min(1e-30)
    return output,target,{'computed_main_qk_pairs':int(valid.sum()),
        'allocated_gather_slots':valid.numel(),'padded_gather_slots':int((~valid).sum())}

