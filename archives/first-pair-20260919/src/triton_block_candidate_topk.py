"""Fused coarse ranking / gathered fine scoring for a KNOWN minmax shortlist.

The coarse FP32 GEMMs remain O(N^2 D / block). Each query then scores only
(routes + current) * block keys in a single Triton program. No global QK scan.
An engineering baseline, not a novel routing method; first-order gradients are
through selected Q/K/V only, as in the existing hard top-k baseline.
"""
import math
import torch
import triton
import triton.language as tl
from torch import nn
from block_candidate_topk import block_summaries, coarse_scores


@triton.jit
def _fine_select(Q, K, Coarse, Ids, N:tl.constexpr, D:tl.constexpr, NB:tl.constexpr,
                 QG:tl.constexpr, QN:tl.constexpr, QD:tl.constexpr,
                 KG:tl.constexpr, KN:tl.constexpr, KD:tl.constexpr,
                 BLOCK:tl.constexpr, ROUTES:tl.constexpr, OFFSET:tl.constexpr,
                 BD:tl.constexpr, BC:tl.constexpr, BB:tl.constexpr):
    i=tl.program_id(0);g=tl.program_id(1)
    current=(i+OFFSET)//BLOCK
    blocks=tl.arange(0,BB)
    coarse=tl.load(Coarse+(g*N+i)*NB+blocks,mask=blocks<NB,other=-float('inf'))
    coarse=tl.where(blocks<current,coarse,-float('inf'))
    slots=tl.arange(0,BC);bucket=slots//BLOCK;within=slots%BLOCK
    selected_blocks=tl.full((BC,),-1,tl.int32)
    for r in range(ROUTES):
        best=tl.max(coarse,0)
        index=tl.min(tl.where(coarse==best,blocks,2147483647),0)
        index=tl.where(best>-float('inf'),index,-1)
        selected_blocks=tl.where(bucket==r,index,selected_blocks)
        coarse=tl.where(blocks==index,-float('inf'),coarse)
    selected_blocks=tl.where(bucket==ROUTES,current,selected_blocks)
    token=selected_blocks*BLOCK+within-OFFSET
    valid=(slots<(ROUTES+1)*BLOCK)&(selected_blocks>=0)&(token>=0)&(token<=i-2)&(token<N)
    d=tl.arange(0,BD)
    q=tl.load(Q+g*QG+i*QN+d*QD,mask=d<D,other=0).to(tl.float32)
    k=tl.load(K+g*KG+token[:,None]*KN+d[None,:]*KD,mask=valid[:,None]&(d[None,:]<D),other=0).to(tl.float32)
    scores=tl.sum(q[None,:]*(k*(D**-0.5)),1)
    scores=tl.where(valid,scores,-float('inf'))
    tl.store(Ids+(g*N+i)*8,i)
    tl.store(Ids+(g*N+i)*8+1,tl.where(i>=1,i-1,-1))
    for r in range(6):
        best=tl.max(scores,0)
        index=tl.min(tl.where((scores==best)&valid,token,2147483647),0)
        index=tl.where(best>-float('inf'),index,-1)
        tl.store(Ids+(g*N+i)*8+r+2,index)
        scores=tl.where(token==index,-float('inf'),scores)


@torch.no_grad()
def select_fused_block_topk(q,k,block=32,routes=4,method='minmax',offset=0):
    if q.shape!=k.shape or q.ndim!=3 or q.shape[1]<8 or not q.is_cuda:
        raise ValueError('Require CUDA equal [G,N,D] and N>=8')
    if block not in [8,16,32,64] or routes not in [1,2,4] or q.shape[-1]>128 or not 0<=offset<block:
        raise ValueError('Unsupported diagnostic kernel configuration')
    summaries=block_summaries(k,block,offset)
    coarse=coarse_scores(q,summaries,method).contiguous()
    groups,n,d=q.shape
    ids=torch.empty(groups,n,8,dtype=torch.int64,device=q.device)
    _fine_select[(n,groups)](q,k,coarse,ids,n,d,coarse.shape[-1],*q.stride(),*k.stride(),
        block,routes,offset,triton.next_power_of_2(d),triton.next_power_of_2((routes+1)*block),
        triton.next_power_of_2(coarse.shape[-1]),num_warps=8,num_stages=1,enable_fp_fusion=False)
    return ids


class FusedBlockCandidateTopKAttention(nn.Module):
    def __init__(self,block=32,routes=4,method='minmax',offset=0,dropout=0.):
        super().__init__();self.block=block;self.routes=routes;self.method=method;self.offset=offset;self.dropout_p=dropout

    def forward(self,qkv):
        from triton_selected_attention import triton_selected_attention
        b,n,_,h,d=qkv.shape
        q,k,v=[x.permute(0,2,1,3).reshape(b*h,n,d) for x in qkv.unbind(2)]
        ids=select_fused_block_topk(q,k,self.block,self.routes,self.method,self.offset)
        out=triton_selected_attention(q,k,v,ids,self.dropout_p if self.training else 0.)
        return out.reshape(b,h,n,d).permute(0,2,1,3)
