"""Experimental QK + exact small-top-k streaming selector, fixed k=8/local=2.

No score matrix is written to global memory. Arithmetic remains quadratic.
Only BF16/FP16 inputs are supported. Scores are rounded to input dtype before
ranking to match the two-stage candidate-score convention, modulo GEMM rounding.
This is an engineering experiment based on known streaming top-k operations.
"""
import math
import torch
import triton
import triton.language as tl
from chunked_topk_attention import _validate
from triton_selected_attention import TritonTopKAttention, triton_selected_attention


@triton.jit
def _insert(a, ai, b, bi):
    take = (b > a) | ((b == a) & (bi < ai))
    return tl.where(take,b,a),tl.where(take,bi,ai),tl.where(take,a,b),tl.where(take,ai,bi)


@triton.jit
def _stream(Q,K,I,N:tl.constexpr,D:tl.constexpr,QG:tl.constexpr,QN:tl.constexpr,QD:tl.constexpr,
            KG:tl.constexpr,KN:tl.constexpr,KD:tl.constexpr,SCALE:tl.constexpr,
            BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    block=tl.program_id(0);group=tl.program_id(1)
    rows=block*BM+tl.arange(0,BM);dd=tl.arange(0,BD)
    q=tl.load(Q+group*QG+rows[:,None]*QN+dd[None,:]*QD,
              (rows[:,None]<N)&(dd[None,:]<D),0)
    b0=tl.full((BM,),-float('inf'),tl.float32);b1=b0;b2=b0;b3=b0;b4=b0;b5=b0
    i0=tl.full((BM,),2147483647,tl.int32);i1=i0;i2=i0;i3=i0;i4=i0;i5=i0
    prefix=tl.minimum((block+1)*BM,N)
    for first in range(0,tl.cdiv(prefix,BK)):
        keys=first*BK+tl.arange(0,BK)
        k=tl.load(K+group*KG+keys[:,None]*KN+dd[None,:]*KD,
                  (keys[:,None]<N)&(dd[None,:]<D),0)
        # Preserve the original scaled-key rounding before the tensor-core product.
        scaled=(k.to(tl.float32)*SCALE).to(k.dtype)
        score=tl.dot(q,tl.trans(scaled),allow_tf32=False)
        score=score.to(q.dtype).to(tl.float32)
        score=tl.where((rows[:,None]<N)&(keys[None,:]<=rows[:,None]-2),score,-float('inf'))
        for rank in tl.static_range(6):
            offset=tl.argmax(score,1,tie_break_left=True)
            candidate=tl.max(score,1);ci=first*BK+offset
            b0,i0,candidate,ci=_insert(b0,i0,candidate,ci)
            b1,i1,candidate,ci=_insert(b1,i1,candidate,ci)
            b2,i2,candidate,ci=_insert(b2,i2,candidate,ci)
            b3,i3,candidate,ci=_insert(b3,i3,candidate,ci)
            b4,i4,candidate,ci=_insert(b4,i4,candidate,ci)
            b5,i5,candidate,ci=_insert(b5,i5,candidate,ci)
            score=tl.where(tl.arange(0,BK)[None,:]==offset[:,None],-float('inf'),score)
    base=I+(group*N+rows)*8
    tl.store(base,rows,rows<N);tl.store(base+1,tl.where(rows>0,rows-1,-1),rows<N)
    tl.store(base+2,tl.where(b0 != -float('inf'),i0,-1),rows<N)
    tl.store(base+3,tl.where(b1 != -float('inf'),i1,-1),rows<N)
    tl.store(base+4,tl.where(b2 != -float('inf'),i2,-1),rows<N)
    tl.store(base+5,tl.where(b3 != -float('inf'),i3,-1),rows<N)
    tl.store(base+6,tl.where(b4 != -float('inf'),i4,-1),rows<N)
    tl.store(base+7,tl.where(b5 != -float('inf'),i5,-1),rows<N)


@torch.no_grad()
def select_streaming_topk(q,k,budget=8,local=2,query_chunk=1024):
    _validate(q,k)
    if not q.is_cuda or q.dtype not in (torch.float16,torch.bfloat16):
        raise ValueError('Streaming selector requires CUDA BF16/FP16; no fallback')
    if budget!=8 or local!=2 or q.shape[1]<8 or q.shape[2]>128:
        raise ValueError('Prototype supports k8/local2, N>=8, D<=128 only')
    g,n,d=q.shape;ids=torch.empty(g,n,8,device=q.device,dtype=torch.long)
    _stream[(triton.cdiv(n,64),g)](q,k,ids,n,d,*q.stride(),*k.stride(),1./math.sqrt(d),
                                  64,128,max(16,triton.next_power_of_2(d)),num_warps=4,num_stages=1)
    return ids


class StreamingTopKAttention(TritonTopKAttention):
    def forward(self,qkv):
        batch,length,three,heads,dim=qkv.shape
        if three!=3:raise ValueError('Expected [B,N,3,H,D]')
        q,k,v=[z.permute(0,2,1,3).reshape(batch*heads,length,dim) for z in qkv.unbind(2)]
        ids=select_streaming_topk(q,k,self.k,self.local,self.query_chunk)
        out=triton_selected_attention(q,k,v,ids,self.dropout_p if self.training else 0.)
        return out.reshape(batch,heads,length,dim).permute(0,2,1,3)
