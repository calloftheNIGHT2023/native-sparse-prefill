"""Query-position attention interventions; computes both paths, no speed claim."""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import run_flashmoba_realtext_precision as base

def dense_attention(q,k,v,scale):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        y=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],
            is_causal=True,dropout_p=0.,scale=scale,enable_gqa=True)
    return y[0].transpose(0,1)

def phase_attention(q,k,v,scale,condition,suffix_start,span=None):
    n=q.shape[0]
    assert 0<suffix_start<n
    if condition=='DD':return dense_attention(q,k,v,scale)
    sparse=base.sparse(q,k,v,128,16,'fp32',scale)
    if condition=='SS':return sparse
    dense=dense_attention(q,k,v,scale)
    if condition=='DS':a,b=0,suffix_start
    elif condition=='SD':a,b=suffix_start,n
    else:
        assert condition in ['target_D','sham_D'] and span is not None
        a,b=span;assert 0<=a<b<=suffix_start
    out=sparse.clone();out[a:b]=dense[a:b]
    return out
