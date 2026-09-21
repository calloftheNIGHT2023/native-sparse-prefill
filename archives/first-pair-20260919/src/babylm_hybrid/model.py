"""Full-parameter CPU reference for the first D/E comparison.

Random initialization only. Explicitly omits MoE, PLE, GR and MTP. The GDN
implementation is pinned to the installed Transformers reference; segmented
calls reset both convolution and recurrent state. No persistent cache, optimized
sparse kernel, training throughput or full-Qwen4 equivalence is claimed.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

from .config import HybridConfig
from .attention import GlobalAttention


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        value = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
                * self.weight).to(x.dtype)


class SigmoidGatedNorm(RMSNorm):
    def forward(self, x, gate):
        return super().forward(x) * gate.sigmoid()


def contiguous_segments(labels):
    """Contiguous nonnegative labels; repeats after another label are new runs."""
    ids = labels.tolist()
    i = 0
    while i < len(ids):
        if ids[i] < 0:
            i += 1
            continue
        end = i + 1
        while end < len(ids) and ids[end] == ids[i]:
            end += 1
        yield i, end
        i = end


class SegmentedGDN(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        reference_cfg = Qwen3NextConfig(
            hidden_size=cfg.hidden_size, linear_num_key_heads=cfg.gdn_key_heads,
            linear_num_value_heads=cfg.gdn_value_heads, linear_key_head_dim=cfg.gdn_key_head_dim,
            linear_value_head_dim=cfg.gdn_value_head_dim, linear_conv_kernel_dim=cfg.gdn_conv_kernel,
            rms_norm_eps=cfg.norm_eps,
        )
        self.core = Qwen3NextGatedDeltaNet(reference_cfg, layer_idx)
        self.core.norm = SigmoidGatedNorm(cfg.gdn_value_head_dim, cfg.norm_eps)

    def forward(self, x, segment_ids):
        rows = []
        for b in range(x.shape[0]):
            row = torch.zeros_like(x[b])
            for start, end in contiguous_segments(segment_ids[b]):
                # No cache is supplied: BOTH causal-convolution and recurrent state reset.
                value = self.core(x[b:b+1, start:end], cache_params=None)
                row = row.index_copy(0, torch.arange(start, end, device=x.device), value[0])
            rows.append(row)
        return torch.stack(rows)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class HybridLayer(nn.Module):
    def __init__(self, cfg, layer_idx, kind):
        super().__init__()
        self.kind = kind
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.mixer = SegmentedGDN(cfg, layer_idx) if kind == "gdn" else GlobalAttention(cfg)
        self.post_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, segment_ids, mode, loss_mask):
        value = self.input_norm(x)
        if self.kind == "gdn":
            mixed = self.mixer(value, segment_ids)
            aux, stats = value.sum() * 0.0, None
        else:
            mixed, aux, stats = self.mixer(value, segment_ids, mode=mode, loss_mask=loss_mask)
        x = x + mixed
        x = x + self.ffn(self.post_norm(x))
        x = x * (segment_ids >= 0).unsqueeze(-1)
        return x, aux, stats


@dataclass
class HybridOutput:
    logits: torch.Tensor
    lm_loss: torch.Tensor
    aux_loss: torch.Tensor
    loss: torch.Tensor
    token_loss_count: int
    attention_stats: list


class HybridLM(nn.Module):
    def __init__(self, cfg: HybridConfig, mode="dense"):
        super().__init__()
        if mode not in ("dense", "sparse"):
            raise ValueError("Only the baseline-first D/E modes are implemented")
        self.cfg, self.mode = cfg, mode
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(HybridLayer(cfg, i, kind) for i, kind in enumerate(cfg.layer_types))
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.untied_output = None if cfg.tie_word_embeddings else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._initialize)

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def initialize_indexers(self):
        for layer in self.layers:
            if layer.kind == "global":
                layer.mixer.initialize_indexer()
                layer.mixer.indexer.apply(self._initialize)

    def forward(self, input_ids, segment_ids=None, loss_mask=None, aux_weight=0.0):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be nonempty [batch, sequence]")
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        if segment_ids.shape != input_ids.shape or bool((segment_ids < -1).any()):
            raise ValueError("segment_ids must align with input IDs; -1 denotes padding")
        valid = segment_ids >= 0
        # LM queries predict only the next token in the SAME contiguous segment.
        query_mask = torch.zeros_like(valid)
        query_mask[:, :-1] = valid[:, :-1] & valid[:, 1:] & (segment_ids[:, :-1] == segment_ids[:, 1:])
        if loss_mask is not None:
            if loss_mask.shape != input_ids.shape:
                raise ValueError("loss_mask shape mismatch")
            query_mask &= loss_mask.bool()
        x = self.embedding(input_ids) * valid.unsqueeze(-1)
        auxiliary, stats = [], []
        for layer in self.layers:
            x, aux, stat = layer(x, segment_ids, self.mode, query_mask)
            if layer.kind == "global":
                auxiliary.append(aux)
                stats.append(stat)
        hidden = self.final_norm(x)
        weight = self.embedding.weight if self.untied_output is None else self.untied_output.weight
        logits = F.linear(hidden, weight)
        targets = torch.zeros_like(input_ids)
        targets[:, :-1] = input_ids[:, 1:]
        count = int(query_mask.sum())
        lm_loss = F.cross_entropy(logits[query_mask], targets[query_mask]) if count else logits.sum() * 0.0
        aux_loss = torch.stack(auxiliary).mean() if auxiliary else logits.sum() * 0.0
        # A disabled auxiliary objective must not create zero indexer gradients
        # that would accidentally trigger AdamW decay on an LM-only update.
        total_loss = lm_loss if aux_weight == 0 else lm_loss + aux_weight * aux_loss
        return HybridOutput(logits, lm_loss, aux_loss, total_loss, count, stats)


def build_model(cfg, mode, backbone_seed, indexer_seed):
    """CPU factory leaves caller RNG unchanged and separates shared/indexer RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(backbone_seed)
        model = HybridLM(cfg, mode=mode)
    if mode == "sparse":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(indexer_seed)
            model.initialize_indexers()
    return model


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    indexer = sum(p.numel() for n, p in model.named_parameters() if ".indexer." in n)
    return {"total": total, "backbone": total-indexer, "indexer": indexer,
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
