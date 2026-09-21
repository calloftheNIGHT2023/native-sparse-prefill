"""Experimental fused selected-edge aggregation; requires CUDA and Triton.

The exact selector is unchanged and still quadratic. This kernel is an engineering
baseline, not an original attention algorithm. First-order gradients only; dK/dV
use FP32 atomics and are not bitwise deterministic. No padding or GQA interface.
"""
import math
import torch
from torch import nn
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
from chunked_topk_attention import _validate, select_causal_topk


@triton.jit
def _forward(Q, K, V, I, KEEP, O, N: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
             QG: tl.constexpr, QN: tl.constexpr, QD: tl.constexpr,
             KG: tl.constexpr, KN: tl.constexpr, KD: tl.constexpr,
             VG: tl.constexpr, VN: tl.constexpr, VD: tl.constexpr,
             SCALE: tl.constexpr, DROP: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0); group = tl.program_id(1)
    ss = tl.arange(0, BS); dd = tl.arange(0, BD)
    ids = tl.load(I + (group * N + row) * S + ss, ss < S, -1)
    valid = (ss < S) & (ids >= 0) & (ids <= row)
    q = tl.load(Q + group * QG + row * QN + dd * QD, dd < D, 0).to(tl.float32)
    k = tl.load(K + group * KG + ids[:, None] * KN + dd[None, :] * KD,
                valid[:, None] & (dd[None, :] < D), 0).to(tl.float32)
    v = tl.load(V + group * VG + ids[:, None] * VN + dd[None, :] * VD,
                valid[:, None] & (dd[None, :] < D), 0).to(tl.float32)
    score = tl.sum(q[None, :] * (k * SCALE), 1)
    score = tl.where(valid, score, -float('inf'))
    ex = tl.exp(score - tl.max(score, 0))
    p = ex / tl.sum(ex, 0)
    if DROP > 0:
        keep = tl.load(KEEP + (group * N + row) * S + ss, ss < S, 0).to(tl.float32)
        p = p * keep / (1.0 - DROP)
    output = tl.sum(p[:, None] * v, 0)
    tl.store(O + (group * N + row) * D + dd, output, dd < D)


@triton.jit
def _backward(Q, K, V, I, KEEP, DO, DQ, DK, DV,
              N: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
              QG: tl.constexpr, QN: tl.constexpr, QD: tl.constexpr,
              KG: tl.constexpr, KN: tl.constexpr, KD: tl.constexpr,
              VG: tl.constexpr, VN: tl.constexpr, VD: tl.constexpr,
              SCALE: tl.constexpr, DROP: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0); group = tl.program_id(1)
    ss = tl.arange(0, BS); dd = tl.arange(0, BD)
    ids = tl.load(I + (group * N + row) * S + ss, ss < S, -1)
    valid = (ss < S) & (ids >= 0) & (ids <= row)
    mask = valid[:, None] & (dd[None, :] < D)
    q = tl.load(Q + group * QG + row * QN + dd * QD, dd < D, 0).to(tl.float32)
    k = tl.load(K + group * KG + ids[:, None] * KN + dd[None, :] * KD, mask, 0).to(tl.float32)
    v = tl.load(V + group * VG + ids[:, None] * VN + dd[None, :] * VD, mask, 0).to(tl.float32)
    do = tl.load(DO + (group * N + row) * D + dd, dd < D, 0).to(tl.float32)
    score = tl.sum(q[None, :] * (k * SCALE), 1)
    score = tl.where(valid, score, -float('inf'))
    ex = tl.exp(score - tl.max(score, 0)); p = ex / tl.sum(ex, 0)
    multiplier = tl.full((BS,), 1.0, tl.float32)
    if DROP > 0:
        keep = tl.load(KEEP + (group * N + row) * S + ss, ss < S, 0).to(tl.float32)
        multiplier = keep / (1.0 - DROP)
    dp = tl.sum(do[None, :] * v, 1) * multiplier
    ds = p * (dp - tl.sum(p * dp, 0))
    dq = tl.sum(ds[:, None] * (k * SCALE), 0)
    tl.store(DQ + (group * N + row) * D + dd, dq, dd < D)
    dk = ds[:, None] * (q[None, :] * SCALE)
    dv = (p * multiplier)[:, None] * do[None, :]
    offset = (group * N + ids[:, None]) * D + dd[None, :]
    tl.atomic_add(DK + offset, dk, mask, sem='relaxed')
    tl.atomic_add(DV + offset, dv, mask, sem='relaxed')


def _launch_args(q, k, v, ids, dropout):
    return (q.shape[1], q.shape[2], ids.shape[2], *q.stride(), *k.stride(), *v.stride(),
            1.0 / math.sqrt(q.shape[2]), dropout, triton.next_power_of_2(ids.shape[2]),
            triton.next_power_of_2(q.shape[2]))


class _TritonSelected(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ids, keep, dropout):
        output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        _forward[(q.shape[1], q.shape[0])](q, k, v, ids, keep, output,
            *_launch_args(q, k, v, ids, dropout), num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(q, k, v, ids, keep); ctx.dropout = dropout
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        q, k, v, ids, keep = ctx.saved_tensors
        grad = grad.contiguous()
        dq = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        dk = torch.zeros_like(dq); dv = torch.zeros_like(dq)
        _backward[(q.shape[1], q.shape[0])](q, k, v, ids, keep, grad, dq, dk, dv,
            *_launch_args(q, k, v, ids, ctx.dropout), num_warps=4, enable_fp_fusion=False)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def triton_selected_attention(q, k, v, ids, dropout_p=0., dropout_keep=None):
    _validate(q, k, v)
    if not q.is_cuda or q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError('CUDA FP32/FP16/BF16 required; no fallback')
    if ids.ndim != 3 or ids.shape[:2] != q.shape[:2] or ids.dtype != torch.long or ids.device != q.device:
        raise ValueError('Indices must be CUDA int64 [G,N,S]')
    if not 1 <= ids.shape[2] <= 64 or not 1 <= q.shape[2] <= 256 or not 0 <= dropout_p < 1:
        raise ValueError('Require slots <=64, head dim <=256, and valid dropout')
    # Caller must supply unique causal indices and at least one valid slot per row.
    ids = ids.contiguous()
    if dropout_keep is None:
        keep = torch.rand(ids.shape, device=q.device) >= dropout_p if dropout_p else torch.empty(0, device=q.device, dtype=torch.bool)
    else:
        if dropout_keep.shape != ids.shape or dropout_keep.dtype != torch.bool or dropout_keep.device != q.device:
            raise ValueError('Invalid fixed dropout mask')
        keep = dropout_keep.contiguous()
    return _TritonSelected.apply(q, k, v, ids, keep, float(dropout_p))


class TritonTopKAttention(nn.Module):
    def __init__(self, k=8, local=2, dropout=.1, query_chunk=1024):
        super().__init__()
        self.k, self.local, self.dropout_p, self.query_chunk = k, local, dropout, query_chunk

    def forward(self, qkv):
        batch, length, three, heads, dim = qkv.shape
        if three != 3:
            raise ValueError('Expected [B,N,3,H,D]')
        q, k, v = [z.permute(0, 2, 1, 3).reshape(batch * heads, length, dim) for z in qkv.unbind(2)]
        ids = select_causal_topk(q, k, self.k, self.local, self.query_chunk)
        out = triton_selected_attention(q, k, v, ids, self.dropout_p if self.training else 0.)
        return out.reshape(batch, heads, length, dim).permute(0, 2, 1, 3)
