"""CPU correctness reference for QSA-style selection and supervision coverage.

Dense masks/score tensors are intentional here: this is NOT a fast sparse kernel,
the full Qwen architecture, or a source of training/prefill speedup claims.
Block pooling, RMS normalization, partial RoPE, summed ReLU index heads, complete
causal blocks and incomplete tails follow the QSA description. Dimensions shrink.
Document boundaries restart blocks; repeated document labels are separate segments.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class Layout:
    block_tokens: torch.Tensor  # [blocks, r]
    visible_blocks: torch.Tensor  # [tokens, blocks]
    tail_mask: torch.Tensor  # [tokens, tokens]
    causal_mask: torch.Tensor  # [tokens, tokens], within contiguous document
    positions: torch.Tensor


def move_layout(layout,device):
    return Layout(*(getattr(layout,name).to(device) for name in Layout.__dataclass_fields__))


def make_layout(doc_ids, block_size=4):
    if block_size < 1 or len(doc_ids) < 1:
        raise ValueError('Need a positive block size and a nonempty document list')
    labels = list(doc_ids)
    n = len(labels)
    blocks, starts = [], [0]
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            starts.append(i)
    ends = starts[1:] + [n]
    causal = torch.zeros(n, n, dtype=torch.bool)
    tails = torch.zeros_like(causal)
    positions = torch.empty(n, dtype=torch.long)
    segments = torch.empty(n, dtype=torch.long)
    for seg, (start, end) in enumerate(zip(starts, ends)):
        for p in range(start, end):
            causal[p, start:p + 1] = True
            positions[p] = p - start
            segments[p] = seg
            tail_start = start + ((p - start + 1) // block_size) * block_size
            tails[p, tail_start:p + 1] = True
        for p in range(start, end - block_size + 1, block_size):
            blocks.append(list(range(p, p + block_size)))
    bt = torch.tensor(blocks, dtype=torch.long).reshape(-1, block_size)
    visible = torch.zeros(n, len(bt), dtype=torch.bool)
    if len(bt):
        visible = ((segments[:, None] == segments[bt[:, 0]][None, :]) &
                   (torch.arange(n)[:, None] >= bt[:, -1][None, :]))
    return Layout(bt, visible, tails, causal, positions)


def partial_rope(x, positions, rotary_dim):
    if rotary_dim == 0:
        return x
    if rotary_dim % 2 or rotary_dim > x.shape[-1]:
        raise ValueError('Rotary dimension must be even and no larger than head size')
    freq = 10000.0 ** (-torch.arange(0, rotary_dim, 2, dtype=x.dtype, device=x.device) / rotary_dim)
    theta = positions.to(x.dtype)[:, None] * freq[None, :]
    while theta.ndim < x.ndim:
        theta = theta.unsqueeze(1)
    a, b = x[..., :rotary_dim:2], x[..., 1:rotary_dim:2]
    rot = torch.stack((a * theta.cos() - b * theta.sin(),
                       a * theta.sin() + b * theta.cos()), dim=-1).flatten(-2)
    return torch.cat((rot, x[..., rotary_dim:]), dim=-1)


def rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


class QSAIndexer(nn.Module):
    def __init__(self, model_dim=16, heads=2, head_dim=8, rotary_dim=4):
        super().__init__()
        self.heads, self.head_dim, self.rotary_dim = heads, head_dim, rotary_dim
        self.query = nn.Linear(model_dim, heads * head_dim, bias=False)
        self.key = nn.Linear(model_dim, head_dim, bias=False)

    def forward(self, hidden, layout):
        hidden = hidden.detach()  # auxiliary objective updates indexer only
        q = rms(self.query(hidden).reshape(len(hidden), self.heads, self.head_dim))
        projected_keys = self.key(hidden)
        k = rms(projected_keys[layout.block_tokens].mean(1))
        q = partial_rope(q, layout.positions, self.rotary_dim)
        k = partial_rope(k, layout.positions[layout.block_tokens[:, 0]], self.rotary_dim)
        return torch.relu(torch.einsum('thd,bd->thb', q, k)).sum(1)


def select_blocks(scores, visible, k):
    if k < 1:
        raise ValueError('At least one complete block must be allowed')
    selected = torch.zeros_like(visible)
    if scores.shape[-1] == 0:
        return selected
    # Stable ties prefer earlier blocks and are reproducible across CPU runs.
    order = scores.detach().masked_fill(~visible, -torch.inf).argsort(
        dim=-1, descending=True, stable=True)
    selected.scatter_(1, order[:, :min(k, scores.shape[-1])], True)
    return selected & visible


def sample_outside(selected, visible, count, generator):
    if count < 0:
        raise ValueError('Probe count must be nonnegative')
    outside = visible & ~selected
    probes = torch.zeros_like(visible)
    if count == 0 or visible.shape[-1] == 0:
        return probes
    random_scores = torch.rand(visible.shape, generator=generator).to(visible.device)
    order = random_scores.masked_fill(~outside, -1).argsort(dim=-1, descending=True)
    probes.scatter_(1, order[:, :min(count, visible.shape[-1])], True)
    return probes & outside  # no replacement and no invalid padded selections


def expand_blocks(block_mask, layout, include_tail=True):
    n = block_mask.shape[0]
    mask = layout.tail_mask.clone() if include_tail else torch.zeros(n, n, dtype=torch.bool, device=block_mask.device)
    if len(layout.block_tokens):
        mask[:, layout.block_tokens.flatten()] |= block_mask.repeat_interleave(
            layout.block_tokens.shape[1], dim=-1)
    return mask & layout.causal_mask


def teacher_block_distribution(q, k, candidate_blocks, layout):
    """Detached QK-only teacher on candidate tokens plus the causal incomplete tail.

    q/k shape [tokens, heads, dim]. No V aggregation is performed. This correctness
    reference computes dense QK before masking; this cost must not be advertised
    as the future gathered probe branch's actual cost. Multi-head normalization
    and max-pooling follow QSA. Subset normalization is not globally unbiased.
    """
    with torch.no_grad():
        mask = expand_blocks(candidate_blocks, layout)
        logits = torch.einsum('thd,shd->ths', q.detach(), k.detach()) / math.sqrt(q.shape[-1])
        logits = logits.masked_fill(~mask[:, None, :], -torch.inf)
        nonempty = mask.any(1)
        logits[~nonempty] = 0
        probs = logits.softmax(-1).mean(1) * mask
        if len(layout.block_tokens) == 0:
            return probs.new_zeros((len(q), 0))
        pooled = probs[:, layout.block_tokens].amax(-1) * candidate_blocks
        return pooled / pooled.sum(-1, keepdim=True).clamp_min(1e-30)


def subset_kl(scores, target, support):
    """Mean KL over queries with >=1 supported block; empty queries contribute zero."""
    valid = support.any(-1)
    if not valid.any():
        return scores.sum() * 0
    mask = support[valid]
    logp = scores[valid].masked_fill(~mask, -torch.inf).log_softmax(-1)
    logp = torch.where(mask, logp, torch.zeros_like(logp))
    p = target[valid].detach() * mask
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-30)
    return (p * (p.clamp_min(1e-30).log() - logp)).sum(-1).mean()


def sparse_core(q, k, v, selected, layout):
    """Gathered per-query reference: only selected blocks plus tail enter output."""
    mask = expand_blocks(selected, layout)
    out = []
    for i in range(len(q)):
        ids = mask[i].nonzero().flatten()
        if len(ids) == 0:
            raise ValueError('Empty core attention support')
        logits = torch.einsum('hd,shd->hs', q[i], k[ids]) / math.sqrt(q.shape[-1])
        out.append(torch.einsum('hs,shd->hd', logits.softmax(-1), v[ids]))
    return torch.stack(out)
