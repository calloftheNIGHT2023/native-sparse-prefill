"""Known dense-suffix diagnostic on K32 prefixes; no answer-location input."""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch.nn.attention.bias import causal_lower_right
import run_flashmoba_realtext_precision as base

def dense_suffix(q,k,v,scale,start):
    assert 0<start<q.shape[0]==k.shape[0]==v.shape[0]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        y=F.scaled_dot_product_attention(q[start:].transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],attn_mask=causal_lower_right(q.shape[0]-start,k.shape[0]),dropout_p=0.,scale=scale,enable_gqa=True)
    return y[0].transpose(0,1)

def sparse_prefix_dense_suffix(q,k,v,scale,start):
    # Diagnostic implementation retains discarded sparse suffix computation.
    # Measure total implementation cost later; no assumed speed benefit.
    y=base.sparse(q,k,v,128,32,'fp32',scale)
    y[start:]=dense_suffix(q,k,v,scale,start)
    return y
