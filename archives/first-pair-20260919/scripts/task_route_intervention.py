"""Privileged diagnostic only: replace blocks at question tokens, same K budget."""
import math
import torch
import run_flashmoba_realtext_precision as base

def metadata_from_ids(ids,nb):
 n,h,k=ids.shape
 assert ids.dtype==torch.int32
 valid=ids>=0
 heads=torch.arange(h,device=ids.device)[None,:,None].expand(n,h,k)
 queries=torch.arange(n,device=ids.device)[:,None,None].expand(n,h,k)
 group=heads[valid]*nb+ids[valid].long()
 codes=group*n+queries[valid]
 sorted_codes=torch.sort(codes).values
 counts=torch.bincount(group,minlength=h*nb).to(torch.int32)
 offsets=torch.cat([torch.zeros(1,device=ids.device,dtype=torch.int64),counts.long().cumsum(0)[:-1]])
 return offsets.reshape(1,h,nb),counts.reshape(1,h,nb),(sorted_codes%n).to(torch.int32)

def replace_question_blocks(ids,start,forced,block):
 n,h,k=ids.shape;assert len(set(forced))==len(forced) and 0<len(forced)<k-1
 assert max(forced)<start//block and start<n
 tail=ids[start:];assert bool((tail>=0).all())
 local=(torch.arange(start,n,device=ids.device)//block).to(torch.int32)[:,None,None].expand(-1,h,1)
 force=torch.tensor(forced,device=ids.device,dtype=torch.int32)[None,None,:].expand(n-start,h,-1)
 excluded=(tail==local)|(tail[...,None]==force[...,None,:]).any(-1)
 ranks=torch.arange(k,device=ids.device)[None,None,:].expand_as(tail)
 keep=torch.argsort(torch.where(excluded,k+ranks,ranks),dim=-1)[...,:k-1-len(forced)]
 newtail=torch.cat([local,force,torch.gather(tail,-1,keep)],dim=-1)
 assert bool((newtail.sort(-1).values[...,1:]!=newtail.sort(-1).values[...,:-1]).all())
 result=ids.clone();result[start:]=newtail
 assert torch.equal((result>=0).sum(-1),(ids>=0).sum(-1))
 assert bool((result<=torch.arange(n,device=ids.device)[:,None,None]//block).all())
 assert torch.equal(result[:start],ids[:start])
 return result

def routed_attention(q,k,v,scale,mode='native',start=None,forced=None,target=None,telemetry=None,return_ids=False):
 n,h,d=q.shape;block=128;topk=16;nb=math.ceil(n/block)
 cu=torch.tensor([0,n],device=q.device,dtype=torch.int32);cm=torch.tensor([0,nb],device=q.device,dtype=torch.int32)
 inp=k.float();means=torch.zeros((nb,k.shape[1],d),device=k.device,dtype=torch.float32)
 base.mean_pool_kernel.fn[(nb,1,k.shape[1])](inp,means,d,block,cu,cm,inp.stride(0),inp.stride(1),means.stride(0),means.stride(1),kBlockN=32,num_warps=4,num_stages=3)
 offsets,counts,indices,values,allids=base.flash_moba_cuda.moba_fused_topk(q,means.to(k.dtype),cu,cu,cm,n,n,topk,block,True)
 ids=allids[...,:topk].contiguous()
 if telemetry is not None and target is not None:
  t=torch.tensor(target,device=q.device);coverage=(ids[-1,:, :,None]==t[None,None,:]).any(1).float().mean()
  telemetry.append(dict(target_block_recall_at_final_query=float(coverage),target_blocks=len(target)))
 if mode=='native':
  indices=base.flash_moba_cuda.varlen_sort(offsets.flatten(),(offsets+counts).flatten(),indices)
 else:
  if mode=='forced':ids=replace_question_blocks(ids,start,forced,block)
  else:assert mode=='rebuild'
  offsets,counts,indices=metadata_from_ids(ids,nb)
 out=base.flash_moba_attn_varlen_func(q,k,v,cu,cu,n,n,offsets,counts,indices,base.decide_lg_block_m(topk,block,n,True),block,dropout_p=0.,softmax_scale=scale,causal=True)
 return (out,ids) if return_ids else out
