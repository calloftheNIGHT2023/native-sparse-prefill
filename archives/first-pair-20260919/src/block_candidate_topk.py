"""Known block summaries as a cheap shortlist before exact token top-k.

Diagnostic baseline, not an original algorithm or an optimized GPU kernel.
Mean summaries are MoBA-like; coordinate min/max summaries are Quest-like.
Unlike those methods, the final support here stays at 2 local + 6 remote tokens.
All queries are routed (prefill), and only completed blocks enter coarse ranking.
The current block is searched causally without using its pooled future keys.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def block_summaries(k, block, offset=0):
    groups, length, dim = k.shape
    if not 0 <= offset < block: raise ValueError('offset must be in [0, block)')
    padded = F.pad(k.float(), (0, 0, offset, (-(length + offset)) % block))
    tiles = padded.reshape(groups, -1, block, dim)
    positions = torch.arange(padded.shape[1], device=k.device) - offset
    valid = ((positions >= 0) & (positions < length)).reshape(1, -1, block, 1)
    return (tiles.sum(-2) / valid.sum(-2),
            tiles.masked_fill(~valid, torch.inf).amin(-2),
            tiles.masked_fill(~valid, -torch.inf).amax(-2))


def coarse_scores(q, summaries, method):
    mean, lo, hi = summaries
    q = q.float() / math.sqrt(q.shape[-1])
    if method == 'mean':
        return torch.bmm(q, mean.transpose(1, 2))
    if method == 'minmax':
        return (torch.bmm(q.clamp_min(0), hi.transpose(1, 2))
                + torch.bmm(q.clamp_max(0), lo.transpose(1, 2)))
    raise ValueError(method)


@torch.no_grad()
def select_block_topk(q, k, block=16, routes=2, method='mean', query_chunk=64, offset=0):
    if q.ndim != 3 or q.shape != k.shape or block < 1 or routes < 1 or query_chunk < 1:
        raise ValueError('Require equal [G,N,D], positive block/routes/chunk')
    groups, length, dim = q.shape
    summaries = block_summaries(k, block, offset)
    blocks = summaries[0].shape[1]
    take = min(routes, blocks)
    ids = torch.full((groups, length, min(8, length)), -1, device=q.device, dtype=torch.long)
    group = torch.arange(groups, device=q.device)[:, None, None]
    offsets = torch.arange(block, device=q.device)
    block_ids = torch.arange(blocks, device=q.device)
    for first in range(0, length, query_chunk):
        end = min(first + query_chunk, length)
        pos = torch.arange(first, end, device=q.device)
        current = (pos + offset) // block
        coarse = coarse_scores(q[:, first:end], summaries, method)
        coarse.masked_fill_(block_ids[None, None, :] >= current[None, :, None], -torch.inf)
        values, chosen = coarse.topk(take, dim=-1, sorted=False)
        chosen = chosen.masked_fill(~values.isfinite(), -1)
        remote_ids = chosen[..., None] * block + offsets - offset
        remote_ids.masked_fill_(chosen[..., None] < 0, -1)
        current_ids = (current[:, None] * block + offsets - offset)[None].expand(groups, -1, -1)
        candidates = torch.cat([remote_ids.flatten(-2), current_ids], dim=-1)
        allowed = (candidates >= 0) & (candidates <= pos[None, :, None] - 2)
        gathered = k[group, candidates.clamp(0, length - 1)].float()
        scores = (q[:, first:end, None].float() * (gathered / math.sqrt(dim))).sum(-1)
        scores.masked_fill_(~allowed, -torch.inf)
        local_count = min(2, length)
        local_ids = pos[:, None] - torch.arange(local_count, device=q.device)
        ids[:, first:end, :local_count] = local_ids.masked_fill(local_ids < 0, -1)
        slots = ids.shape[-1] - local_count
        if slots:
            count = min(slots, scores.shape[-1])
            selected_scores, where = scores.topk(count, dim=-1, sorted=False)
            selected_ids = candidates.gather(-1, where).masked_fill(~selected_scores.isfinite(), -1)
            ids[:, first:end, local_count:local_count + count] = selected_ids
    return ids


class BlockCandidateTopKAttention(nn.Module):
    def __init__(self, block=16, routes=2, method='mean', query_chunk=64, backend='torch', dropout=0.1, offset=0):
        super().__init__()
        self.block, self.routes, self.method = block, routes, method
        self.query_chunk, self.backend, self.dropout_p = query_chunk, backend, dropout
        self.offset = offset

    def forward(self, qkv):
        batch, length, _, heads, dim = qkv.shape
        q, k, v = [x.permute(0, 2, 1, 3).reshape(batch * heads, length, dim) for x in qkv.unbind(2)]
        ids = select_block_topk(q, k, self.block, self.routes, self.method, self.query_chunk, self.offset)
        if self.backend == 'triton':
            from triton_selected_attention import triton_selected_attention
            out = triton_selected_attention(q, k, v, ids, self.dropout_p if self.training else 0.)
        else:
            from chunked_topk_attention import selected_attention
            out = selected_attention(q, k, v, ids, self.dropout_p if self.training else 0., self.query_chunk)
        return out.reshape(batch, heads, length, dim).permute(0, 2, 1, 3)
