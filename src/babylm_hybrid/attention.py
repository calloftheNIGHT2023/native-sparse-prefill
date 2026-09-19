"""Differentiable dense-score reference for the BabyLM GDN/QSA baseline.

This module preserves a declared QSA training contract, not an unpublished
official training implementation. It materializes full main-attention and
indexer score tensors and must not support claims about sparse training speed.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _normalization_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


class _RMSNorm(nn.Module):
    """Zero-centered gain: the effective multiplicative gain starts at one."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x.to(_normalization_dtype(x))
        gain = 1.0 + self.weight.to(work.dtype)
        return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps) * gain).to(x.dtype)


def _rope(x: torch.Tensor, positions: torch.Tensor, rotary_dim: int, theta: float) -> torch.Tensor:
    if rotary_dim == 0:
        return x
    work_dtype = _normalization_dtype(x)
    exponent = torch.arange(0, rotary_dim, 2, device=x.device, dtype=work_dtype) / rotary_dim
    frequencies = theta ** (-exponent)
    phases = positions.to(work_dtype)[:, None] * frequencies[None, :]
    while phases.ndim < x.ndim:
        phases = phases.unsqueeze(1)
    cosine, sine = phases.cos().to(x.dtype), phases.sin().to(x.dtype)
    half = rotary_dim // 2
    first, second = x[..., :half], x[..., half:rotary_dim]
    return torch.cat((first * cosine - second * sine,
                      second * cosine + first * sine,
                      x[..., rotary_dim:]), dim=-1)


def _segments(labels: torch.Tensor):
    """Yield positive/nonnegative contiguous runs; -1 breaks every boundary."""
    values = labels.detach().cpu().tolist()
    start = 0
    while start < len(values):
        if values[start] == -1:
            start += 1
            continue
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        yield start, end
        start = end


class _QSAIndexer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = int(cfg.index_query_heads)
        self.dim = int(cfg.index_head_dim)
        self.rotary_dim = int(cfg.index_rotary_dim)
        self.block_size = int(cfg.block_size)
        self.score_scale = float(cfg.index_score_scale)
        self.rope_theta = float(cfg.rope_theta)
        self.q_proj = nn.Linear(cfg.hidden_size, self.heads * self.dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.dim, bias=False)
        self.q_norm = _RMSNorm(self.dim, cfg.norm_eps)
        self.k_norm = _RMSNorm(self.dim, cfg.norm_eps)

    def forward(self, hidden: torch.Tensor):
        hidden = hidden.detach()
        length = hidden.shape[0]
        blocks = length // self.block_size
        query = self.q_norm(self.q_proj(hidden).reshape(length, self.heads, self.dim))
        raw_keys = self.k_proj(hidden)
        # Pool before normalization and rotation. FP32 pooling is deliberate,
        # including when the surrounding numerical reference uses float64.
        pooled = raw_keys[:blocks * self.block_size].reshape(blocks, self.block_size, self.dim).float().mean(1)
        key = self.k_norm(pooled.to(raw_keys.dtype))
        positions = torch.arange(length, device=hidden.device)
        block_starts = torch.arange(blocks, device=hidden.device) * self.block_size
        query = _rope(query, positions, self.rotary_dim, self.rope_theta)
        key = _rope(key, block_starts, self.rotary_dim, self.rope_theta)
        scores = torch.relu(torch.einsum('thd,bd->thb', query, key)).sum(1) * self.score_scale
        visible = (block_starts + self.block_size - 1)[None, :] <= positions[:, None]
        return scores, visible


class GlobalAttention(nn.Module):
    """Gated GQA with optional independent, selected-support QSA indexer.

    ``initialize_indexer`` must be called under a separate RNG stream by the
    model builder. Dense construction/forward never instantiates or runs it.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = int(cfg.hidden_size)
        self.query_heads = int(cfg.global_q_heads)
        self.kv_heads = int(cfg.global_kv_heads)
        self.head_dim = int(cfg.global_head_dim)
        self.rotary_dim = int(cfg.global_rotary_dim)
        self.block_size = int(cfg.block_size)
        self.selected_blocks = int(cfg.selected_complete_blocks)
        self.rope_theta = float(cfg.rope_theta)
        if min(self.hidden_size, self.query_heads, self.kv_heads, self.head_dim, self.block_size, self.selected_blocks) <= 0:
            raise ValueError('Attention dimensions and block budget must be positive')
        if self.query_heads % self.kv_heads:
            raise ValueError('global_q_heads must be divisible by global_kv_heads')
        for name, rotary, head in [('global', self.rotary_dim, self.head_dim),
                                   ('index', int(cfg.index_rotary_dim), int(cfg.index_head_dim))]:
            if rotary < 0 or rotary % 2 or rotary > head:
                raise ValueError(f'{name} rotary dimension must be even and between zero and head dimension')
        if min(int(cfg.index_query_heads), int(cfg.index_head_dim)) <= 0:
            raise ValueError('Indexer dimensions must be positive')
        if float(cfg.index_score_scale) <= 0 or self.rope_theta <= 0 or float(cfg.norm_eps) <= 0:
            raise ValueError('score scale, rope_theta and norm_eps must be positive')
        query_width, kv_width = self.query_heads * self.head_dim, self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(self.hidden_size, query_width, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, kv_width, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, kv_width, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, query_width, bias=False)
        self.out_proj = nn.Linear(query_width, self.hidden_size, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, cfg.norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, cfg.norm_eps)
        self.indexer = None

    def initialize_indexer(self):
        if self.indexer is not None:
            raise RuntimeError('Indexer already initialized; refusing to silently replace its state')
        reference = self.q_proj.weight
        self.indexer = _QSAIndexer(self.cfg).to(device=reference.device, dtype=reference.dtype)

    def _main_qkv(self, x):
        length = x.shape[0]
        positions = torch.arange(length, device=x.device)
        query = self.q_norm(self.q_proj(x).reshape(length, self.query_heads, self.head_dim))
        key = self.k_norm(self.k_proj(x).reshape(length, self.kv_heads, self.head_dim))
        value = self.v_proj(x).reshape(length, self.kv_heads, self.head_dim)
        query = _rope(query, positions, self.rotary_dim, self.rope_theta)
        key = _rope(key, positions, self.rotary_dim, self.rope_theta)
        repeats = self.query_heads // self.kv_heads
        return query, key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1)

    def _sparse_selection(self, scores, visible):
        length, blocks = scores.shape
        chosen = torch.zeros_like(visible)
        if blocks:
            order = scores.detach().masked_fill(~visible, -torch.inf).argsort(dim=-1, descending=True, stable=True)
            chosen.scatter_(1, order[:, :min(blocks, self.selected_blocks)], True)
            chosen &= visible
        positions = torch.arange(length, device=scores.device)
        tail_start = ((positions + 1) // self.block_size) * self.block_size
        mask = (positions[None, :] >= tail_start[:, None]) & (positions[None, :] <= positions[:, None])
        if blocks:
            mask[:, :blocks * self.block_size] |= chosen.repeat_interleave(self.block_size, dim=-1)
        return chosen, mask

    def _segment(self, x, mode, supervised_queries):
        length = x.shape[0]
        query, key, value = self._main_qkv(x)
        stats = {'segments': 1, 'logical_dense_causal_pairs': length * (length + 1) // 2,
                 'allocated_main_score_elements': length * length * self.query_heads,
                 'indexer_score_elements': 0, 'aux_nonempty_support_query_count': 0,
                 'indexer_zero_score_visible_query_count': 0}
        if mode == 'sparse':
            scores, visible = self.indexer(x)
            chosen, mask = self._sparse_selection(scores, visible)
            stats['indexer_score_elements'] = scores.numel()
            stats['indexer_zero_score_visible_query_count'] = int(((scores.masked_fill(~visible, 0).sum(-1) == 0) & visible.any(-1)).sum().detach())
        else:
            mask = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        stats['logical_kept_pairs'] = int(mask.sum().detach())
        logits = torch.einsum('thd,shd->ths', query, key) * (self.head_dim ** -0.5)
        probabilities = logits.masked_fill(~mask[:, None, :], -torch.inf).softmax(-1, dtype=_normalization_dtype(logits)).to(value.dtype)
        attended = torch.einsum('ths,shd->thd', probabilities, value).reshape(length, -1)
        gated = attended * self.gate_proj(x).sigmoid()
        result = self.out_proj(gated)
        if mode == 'dense':
            return result, x.new_zeros(()), stats
        valid = supervised_queries & chosen.any(-1)
        stats['aux_nonempty_support_query_count'] = int(valid.sum().detach())
        if not valid.any():
            return result, scores.sum() * 0, stats
        blocks = chosen.shape[1]
        with torch.no_grad():
            token_teacher = probabilities.detach().to(_normalization_dtype(probabilities)).mean(1)
            block_teacher = token_teacher[:, :blocks * self.block_size].reshape(length, blocks, self.block_size).amax(-1)
            block_teacher = block_teacher * chosen
            teacher = block_teacher / block_teacher.sum(-1, keepdim=True).clamp_min(1e-30)
        supported = chosen[valid]
        log_prediction = scores[valid].to(_normalization_dtype(scores)).masked_fill(~supported, -torch.inf).log_softmax(-1)
        log_prediction = torch.where(supported, log_prediction, torch.zeros_like(log_prediction))
        target = teacher[valid]
        kl_sum = (target * (target.clamp_min(1e-30).log() - log_prediction)).sum()
        return result, kl_sum, stats

    def forward(self, x, segment_ids, mode='dense', loss_mask=None):
        if mode not in ('dense', 'sparse'):
            raise ValueError("mode must be 'dense' or 'sparse'")
        if mode == 'sparse' and self.indexer is None:
            raise RuntimeError('Sparse forward requires initialize_indexer()')
        if x.ndim != 3 or x.shape[-1] != self.hidden_size or segment_ids.shape != x.shape[:2]:
            raise ValueError('Expected x [B,T,H] and segment_ids [B,T]')
        if segment_ids.dtype == torch.bool or segment_ids.is_floating_point():
            raise ValueError('segment_ids must contain integer labels')
        if bool((segment_ids < -1).any()):
            raise ValueError('Only -1 is allowed for padding labels')
        if segment_ids.device != x.device:
            raise ValueError('segment_ids and x must be on the same device')
        if loss_mask is not None and (loss_mask.shape != x.shape[:2] or loss_mask.device != x.device):
            raise ValueError('loss_mask must match [B,T] and x device')
        eligible = segment_ids >= 0
        if loss_mask is not None:
            eligible = eligible & loss_mask.bool()
        denominator = int(eligible.sum().detach())
        output = x * 0
        if mode == 'sparse':
            auxiliary = sum((p.sum() * 0 for p in self.indexer.parameters()), x.new_zeros(()))
        else:
            auxiliary = x.new_zeros(())
        stats = {'backend': 'dense_score_correctness_reference_not_speed_kernel', 'mode': mode,
                 'valid_tokens': int((segment_ids >= 0).sum().detach()), 'aux_loss_query_count': denominator,
                 'segments': 0, 'logical_dense_causal_pairs': 0, 'logical_kept_pairs': 0,
                 'allocated_main_score_elements': 0, 'indexer_score_elements': 0,
                 'aux_nonempty_support_query_count': 0, 'indexer_zero_score_visible_query_count': 0}
        for batch in range(x.shape[0]):
            for start, end in _segments(segment_ids[batch]):
                segment_out, kl_sum, segment_stats = self._segment(x[batch, start:end], mode, eligible[batch, start:end])
                output[batch, start:end] = segment_out
                auxiliary = auxiliary + kl_sum
                for name, value in segment_stats.items():
                    stats[name] += value
        output = output.masked_fill((segment_ids < 0).unsqueeze(-1), 0)
        auxiliary = auxiliary / max(denominator, 1)
        stats['logical_retained_fraction'] = stats['logical_kept_pairs'] / max(stats['logical_dense_causal_pairs'], 1)
        return output, auxiliary, stats
