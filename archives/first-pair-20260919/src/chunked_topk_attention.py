"""Exact causal top-k baseline with bounded query tiles and selected-edge backward.

Known query-chunking/recomputation ideas, NOT a new attention method or a fused
GPU kernel. The selector still evaluates O(N**2 * D) scores. It runs without
autograd and discards each [batch*heads, query_chunk, key_prefix] score tile.
Only Q/K/V, compact indices and (optionally) dropout bits survive for backward.

Interface: equal-length Q/K/V, no padding, no GQA, first-order gradients only.
Floating-point ranking ties need not match torch.topk on a differently sized
matrix. Compact dropout has the same distribution but a different RNG layout
from the historical full-matrix reference; training trajectories are not equal.
"""
import math
import torch
from torch import nn
from torch.autograd.function import once_differentiable


def _validate(q, k, v=None):
    if q.ndim != 3 or k.shape != q.shape or min(q.shape) < 1:
        raise ValueError('Q/K must share nonempty [batch*heads, length, dim] shape')
    if q.device != k.device or q.dtype != k.dtype or not q.is_floating_point():
        raise ValueError('Q/K must share floating dtype and device')
    if v is not None and (v.shape != q.shape or v.dtype != q.dtype or v.device != q.device):
        raise ValueError('V must share Q/K shape, dtype and device')


@torch.no_grad()
def select_causal_topk(q, k, budget=8, local=2, query_chunk=64):
    """Return [G,N,min(budget,N)] int64 indices; -1 denotes an unused slot.

    Local positions are always retained; the remaining slots select the highest
    scores among nonlocal causal keys. For early rows, unused slots stay invalid
    rather than gathering a duplicate local token. No full N*N mask is built.
    """
    _validate(q, k)
    if not 1 <= local <= budget or query_chunk < 1:
        raise ValueError('Require 1 <= local <= budget and query_chunk >= 1')
    groups, length, dim = q.shape
    slots = min(budget, length)
    local_slots = min(local, slots)
    remote_slots = slots - local_slots
    ids = torch.full((groups, length, slots), -1, device=q.device, dtype=torch.long)
    offsets = torch.arange(local_slots, device=q.device)
    # This scaled key tensor is O(G*N*D); it is not retained by autograd.
    scaled_k = k / math.sqrt(dim) if remote_slots else None
    for first in range(0, length, query_chunk):
        end = min(first + query_chunk, length)
        positions = torch.arange(first, end, device=q.device)
        local_ids = positions[:, None] - offsets
        ids[:, first:end, :local_slots] = local_ids.masked_fill(local_ids < 0, -1)
        if remote_slots:
            scores = torch.bmm(q[:, first:end], scaled_k[:, :end].transpose(1, 2))
            keys = torch.arange(end, device=q.device)
            allowed = keys[None, :] <= positions[:, None] - local
            scores.masked_fill_(~allowed, -torch.inf)
            take = min(remote_slots, end)
            values, candidates = scores.topk(take, dim=-1, sorted=False)
            candidates.masked_fill_(~torch.isfinite(values), -1)
            ids[:, first:end, local_slots:local_slots + take] = candidates
    return ids


def _compute_dtype(dtype):
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _selected_chunk(q, k, v, ids, first, end):
    """Temporary selected K/V tensors are bounded by query_chunk * budget."""
    chosen = ids[:, first:end]
    groups = torch.arange(len(q), device=q.device)[:, None, None]
    safe = chosen.clamp_min(0)
    dtype = _compute_dtype(q.dtype)
    query = q[:, first:end].to(dtype)
    keys = k[groups, safe].to(dtype)
    values = v[groups, safe].to(dtype)
    scores = (query[:, :, None] * (keys / math.sqrt(q.shape[-1]))).sum(-1)
    probabilities = scores.masked_fill(chosen < 0, -torch.inf).softmax(-1)
    return query, keys, values, probabilities, safe


def _drop_multiplier(keep, first, end, probability, dtype):
    if not probability:
        return 1.0
    return keep[:, first:end].to(dtype) / (1.0 - probability)


class _SelectedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ids, keep, dropout_p, query_chunk):
        ctx.save_for_backward(q, k, v, ids, keep)
        ctx.dropout_p, ctx.query_chunk = dropout_p, query_chunk
        output = torch.empty_like(q)
        for first in range(0, q.shape[1], query_chunk):
            end = min(first + query_chunk, q.shape[1])
            _, _, values, probs, _ = _selected_chunk(q, k, v, ids, first, end)
            weights = probs * _drop_multiplier(keep, first, end, dropout_p, probs.dtype)
            output[:, first:end] = (weights[..., None] * values).sum(-2).to(q.dtype)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, ids, keep = ctx.saved_tensors
        dtype = _compute_dtype(q.dtype)
        dq, dk, dv = [torch.zeros(q.shape, device=q.device, dtype=dtype) for _ in range(3)]
        dim = q.shape[-1]
        for first in range(0, q.shape[1], ctx.query_chunk):
            end = min(first + ctx.query_chunk, q.shape[1])
            query, keys, values, probs, safe = _selected_chunk(q, k, v, ids, first, end)
            go = grad_output[:, first:end].to(dtype)
            multiplier = _drop_multiplier(keep, first, end, ctx.dropout_p, dtype)
            weights = probs * multiplier
            dprobs = (go[:, :, None] * values).sum(-1) * multiplier
            dscores = probs * (dprobs - (probs * dprobs).sum(-1, keepdim=True))
            dq[:, first:end] = (dscores[..., None] * (keys / math.sqrt(dim))).sum(-2)
            dkeys = dscores[..., None] * (query[:, :, None] / math.sqrt(dim))
            dvalues = weights[..., None] * go[:, :, None]
            # Many queries can address the same key: accumulate, never overwrite.
            scatter_ids = safe.reshape(len(q), -1, 1).expand(-1, -1, dim)
            dk.scatter_add_(1, scatter_ids, dkeys.reshape(len(q), -1, dim))
            dv.scatter_add_(1, scatter_ids, dvalues.reshape(len(q), -1, dim))
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def selected_attention(q, k, v, ids, dropout_p=0.0, query_chunk=64, dropout_keep=None):
    """Compact selected-edge aggregation with recomputing first-order backward.

    ids must be valid unique causal positions per row, with >=1 visible entry;
    use select_causal_topk to construct them. An optional dropout_keep exists for
    independently auditing gradients against an ordinary autograd reference.
    """
    _validate(q, k, v)
    if ids.ndim != 3 or ids.shape[:2] != q.shape[:2] or ids.dtype != torch.long or ids.device != q.device:
        raise ValueError('Invalid index shape, dtype or device')
    if not 0 <= dropout_p < 1 or query_chunk < 1:
        raise ValueError('Invalid dropout or chunk size')
    if dropout_keep is not None:
        if dropout_keep.shape != ids.shape or dropout_keep.dtype != torch.bool or dropout_keep.device != q.device:
            raise ValueError('Invalid fixed dropout mask')
        keep = dropout_keep
    elif dropout_p:
        keep = torch.rand(ids.shape, device=q.device) >= dropout_p
    else:
        keep = torch.empty(0, dtype=torch.bool, device=q.device)
    return _SelectedAttention.apply(q, k, v, ids, keep, float(dropout_p), int(query_chunk))


class ChunkedTopKAttention(nn.Module):
    """Zoology-compatible [B,N,3,H,D] adapter; contains no learned parameters."""
    def __init__(self, k=8, local=2, dropout=0.1, query_chunk=64):
        super().__init__()
        if not 1 <= local <= k or not 0 <= dropout < 1 or query_chunk < 1:
            raise ValueError('Invalid sparse attention configuration')
        self.k, self.local = k, local
        self.dropout_p, self.query_chunk = dropout, query_chunk

    def forward(self, qkv):
        if qkv.ndim != 5 or qkv.shape[2] != 3:
            raise ValueError('Expected [batch,length,3,heads,dim]')
        batch, length, _, heads, dim = qkv.shape
        q, k, v = [x.permute(0, 2, 1, 3).reshape(batch * heads, length, dim) for x in qkv.unbind(2)]
        ids = select_causal_topk(q, k, self.k, self.local, self.query_chunk)
        out = selected_attention(q, k, v, ids, self.dropout_p if self.training else 0.0, self.query_chunk)
        return out.reshape(batch, heads, length, dim).permute(0, 2, 1, 3)


def install_chunked(model, k=8, local=2, query_chunk=64):
    for layer in model.backbone.layers:
        old = layer.sequence_mixer.inner_attn
        replacement = ChunkedTopKAttention(k, local, old.dropout_p, query_chunk)
        replacement.train(old.training)
        layer.sequence_mixer.inner_attn = replacement


def operation_accounting(length, budget, local, query_chunk, groups=1):
    """Analytical pair/slot counts, NOT measured FLOPs, memory peaks or speedup."""
    slots = min(budget, length)
    selected = groups * sum(min(i + 1, slots) for i in range(length))
    tiles = [(min(query_chunk, length - start), min(length, start + query_chunk))
             for start in range(0, length, query_chunk)]
    remote = slots > min(local, slots)
    return dict(selector_dot_products=groups * sum(c * end for c, end in tiles) if remote else 0,
                peak_selector_score_elements=groups * max(c * end for c, end in tiles) if remote else 0,
                selected_valid_edges=selected, allocated_selected_slots=groups * length * slots,
                selector_remains_quadratic=remote,
                interpretation='Pair counts only; include selector and gather/scatter overhead in timing.')
