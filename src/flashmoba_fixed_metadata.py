"""Fixed-length metadata wrapper around unmodified pinned official FlashMoBA kernels.

Engineering adapter, not a new sparse attention algorithm. Caches only lengths and
allocation sizes; K means, routing, sorting and attention recompute for every call.
Use official eager API as the primary reference. Never use with changed lengths.
"""
import math
import torch
from torch import nn
from flash_moba.flash_moba_interface import (
    _wrapped_moba_fused_topk, _wrapped_varlen_sort,
    flash_moba_attn_varlen_func, decide_lg_block_m,
)
from flash_moba.triton_mean_pool import mean_pool_kernel

class FlashMoBAFixedMetadata(nn.Module):
    def __init__(self,lengths,heads,dim,block=128,topk=4,kv_heads=None,device='cuda'):
        super().__init__()
        assert block%64==0 and dim%8==0 and len(lengths)>0 and min(lengths)>0
        self.lengths=tuple(lengths);self.heads=heads;self.kv_heads=kv_heads or heads;self.dim=dim
        self.block=block;self.topk=topk;self.total=sum(lengths);self.maxlen=max(lengths)
        cumulative=[0];blocks=[0]
        for n in lengths:cumulative.append(cumulative[-1]+n);blocks.append(blocks[-1]+math.ceil(n/block))
        self.total_blocks=blocks[-1];self.max_blocks=math.ceil(self.maxlen/block)
        self.register_buffer('cu',torch.tensor(cumulative,dtype=torch.int32,device=device))
        self.register_buffer('cu_blocks',torch.tensor(blocks,dtype=torch.int32,device=device))
        self.lg_m=decide_lg_block_m(topk,block,self.maxlen,True)

    def forward(self,q,k,v):
        assert q.shape==(self.total,self.heads,self.dim)
        assert k.shape==v.shape==(self.total,self.kv_heads,self.dim)
        assert q.dtype==k.dtype==v.dtype and q.device==k.device==v.device==self.cu.device
        means=torch.zeros((self.total_blocks,self.kv_heads,self.dim),dtype=k.dtype,device=k.device)
        mean_pool_kernel[(self.max_blocks,len(self.lengths),self.kv_heads)](
            k,means,self.dim,self.block,self.cu,self.cu_blocks,
            k.stride(0),k.stride(1),means.stride(0),means.stride(1))
        offsets,counts,indices=_wrapped_moba_fused_topk(
            q,means,self.cu,self.cu,self.cu_blocks,self.maxlen,self.maxlen,self.topk,self.block,causal=True)
        indices=_wrapped_varlen_sort(offsets,counts,indices)
        return flash_moba_attn_varlen_func(q,k,v,self.cu,self.cu,self.maxlen,self.maxlen,
            offsets,counts,indices,self.lg_m,self.block,dropout_p=0.,causal=True)
