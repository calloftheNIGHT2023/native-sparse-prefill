"""Trainable fixed local W reference, with the original dense-score allocation.

Each query attends to the most recent K complete B-token blocks and its current
incomplete causal tail. No sink, router, auxiliary objective, or sparse-speed
claim. The original model/attention implementation is unchanged.
"""
from __future__ import annotations

import torch

from .attention import GlobalAttention, _normalization_dtype
from .model import HybridLM


CONDITION = 'local_recent_complete_blocks_plus_causal_tail_v0'
CONTRACT_BUFFER = 'local_attention_contract_v0'


def local_block_selection(length, block_size, selected_blocks, device=None):
    """Return complete-block support and token support, both boolean.

    Positions restart at each contiguous document segment. At query q the
    number of completed blocks is floor((q+1)/B); the tail is [(q+1)//B*B,q].
    """
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (length, block_size, selected_blocks)):
        raise TypeError('Length, block size and budget must be integers')
    if length <= 0 or block_size <= 0 or selected_blocks <= 0:
        raise ValueError('Length, block size and budget must be positive')
    positions = torch.arange(length, device=device)
    completed = (positions + 1) // block_size
    blocks = torch.arange(length // block_size, device=device)
    first = (completed - selected_blocks).clamp_min(0)
    chosen = (blocks[None, :] >= first[:, None]) & (blocks[None, :] < completed[:, None])
    # Complete recent blocks and the incomplete tail form one causal interval.
    mask = (positions[None, :] >= (first * block_size)[:, None]) & (positions[None, :] <= positions[:, None])
    return chosen, mask


class LocalGlobalAttention(GlobalAttention):
    """Same gated GQA parameters as D; no indexer can be instantiated."""

    def initialize_indexer(self):
        raise RuntimeError('Fixed local W has no indexer')

    def _segment(self, x, mode, supervised_queries):
        if mode != 'dense':
            raise ValueError('Internal local reference uses the no-aux dense execution path')
        length = x.shape[0]
        _, mask = local_block_selection(length, self.block_size, self.selected_blocks, x.device)
        query, key, value = self._main_qkv(x)
        logits = torch.einsum('thd,shd->ths', query, key) * self.head_dim ** -0.5
        probabilities = logits.masked_fill(~mask[:, None, :], -torch.inf).softmax(
            -1, dtype=_normalization_dtype(logits)).to(value.dtype)
        attended = torch.einsum('ths,shd->thd', probabilities, value).reshape(length, -1)
        output = self.out_proj(attended * self.gate_proj(x).sigmoid())
        stats = {'segments': 1, 'logical_dense_causal_pairs': length * (length + 1) // 2,
                 'logical_kept_pairs': int(mask.sum().detach()),
                 'allocated_main_score_elements': length * length * self.query_heads,
                 'indexer_score_elements': 0, 'aux_nonempty_support_query_count': 0,
                 'indexer_zero_score_visible_query_count': 0}
        return output, x.new_zeros(()), stats

    def forward(self, x, segment_ids, mode='local', loss_mask=None):
        if mode not in ('dense', 'local'):
            raise ValueError('Fixed local W only accepts local or dense compatibility mode')
        if self.indexer is not None:
            raise RuntimeError('Unexpected indexer in fixed local W')
        output, aux, stats = super().forward(x, segment_ids, mode='dense', loss_mask=loss_mask)
        stats['mode'] = 'local'
        stats['condition'] = CONDITION
        return output, aux, stats


class LocalHybridLM(HybridLM):
    """Original backbone initialization, local mixers, persistent identity tag."""

    def __init__(self, cfg):
        super().__init__(cfg, mode='dense')
        # Replacement construction must not advance the caller or initial-model
        # RNG. Every backbone tensor is copied exactly from its dense counterpart.
        with torch.random.fork_rng(devices=[]):
            for layer in self.layers:
                if layer.kind == 'global':
                    old = layer.mixer
                    local = LocalGlobalAttention(cfg)
                    local.load_state_dict(old.state_dict(), strict=True)
                    layer.mixer = local
        self.mode = 'local'
        self.attention_condition = CONDITION
        self.register_buffer(CONTRACT_BUFFER, torch.tensor(
            [int(cfg.block_size), int(cfg.selected_complete_blocks)], dtype=torch.int64), persistent=True)

    def initialize_indexers(self):
        raise RuntimeError('Fixed local W has no indexer initialization')

    def load_state_dict(self, state_dict, *args, **kwargs):
        marker = state_dict.get(CONTRACT_BUFFER)
        expected = getattr(self, CONTRACT_BUFFER)
        if marker is None or marker.dtype != torch.int64 or not torch.equal(marker.detach().cpu(), expected.detach().cpu()):
            raise RuntimeError('Missing or mismatched fixed-local checkpoint contract')
        return super().load_state_dict(state_dict, *args, **kwargs)


def build_local_model(cfg, mode, backbone_seed, indexer_seed):
    """Signature-compatible with build_model; indexer_seed has no effect.

    ``dense`` is accepted only for the frozen engine's backbone-only optimizer
    compatibility. The returned model and every attention statistic say local.
    The persistent buffer prevents a dense evaluator silently loading W.
    """
    if mode not in ('dense', 'local'):
        raise ValueError('Fixed local builder refuses sparse/indexer mode')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(backbone_seed)
        model = LocalHybridLM(cfg)
    return model
