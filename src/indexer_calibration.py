"""Diagnostic factors for score scale and learned RMS gains, not a new QSA method.

NeMo/vLLM expose learned norm gains; inference score scale alone cannot specify
the original training KL temperature. Preserve the legacy reference separately.
"""
import math
import torch
from torch import nn
from sparse_reference import QSAIndexer, rms, partial_rope


class CalibratedIndexer(QSAIndexer):
    def __init__(self, model_dim, heads, head_dim, rotary_dim, affine=False, scaled=False):
        super().__init__(model_dim, heads, head_dim, rotary_dim)
        self.affine = affine
        self.score_scale = head_dim ** -0.5 if scaled else 1.0
        if affine:
            # Delta parameterization: effective gain is initially exactly one.
            self.query_gain = nn.Parameter(torch.zeros(head_dim))
            self.key_gain = nn.Parameter(torch.zeros(head_dim))

    def forward(self, hidden, layout):
        hidden = hidden.detach()
        q = rms(self.query(hidden).reshape(len(hidden), self.heads, self.head_dim))
        k = rms(self.key(hidden)[layout.block_tokens].mean(1))
        if self.affine:
            q = q * (1 + self.query_gain)
            k = k * (1 + self.key_gain)
        q = partial_rope(q, layout.positions, self.rotary_dim)
        k = partial_rope(k, layout.positions[layout.block_tokens[:, 0]], self.rotary_dim)
        return torch.relu(torch.einsum('thd,bd->thb', q, k)).sum(1) * self.score_scale


def make_indexer(model_dim, cfg):
    args = (model_dim, cfg['index_heads'], cfg['index_head_dim'], cfg['rotary_dim'])
    if 'indexer_calibration' in cfg:
        return CalibratedIndexer(*args, **cfg['indexer_calibration'])
    return QSAIndexer(*args)
