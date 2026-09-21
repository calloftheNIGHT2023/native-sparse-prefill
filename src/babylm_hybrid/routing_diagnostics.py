"""Read-only, same-hidden routing diagnostics for the frozen QSA reference.

This deliberately allocates dense attention scores. It measures quality and
router behavior, never sparse-kernel speed. No model/optimizer calls are made by
this module: the caller supplies ordinary evaluation forwards inside a context.
"""
from __future__ import annotations

import copy
import hashlib
import types
import weakref

import torch

from .attention import GlobalAttention, _normalization_dtype, _rope


POLICIES = ("learned", "prefix", "local", "random")
_ACTIVE = weakref.WeakKeyDictionary()


def select_support(scores, visible, *, block_size, topk, policy="learned",
                   seed=0, layer_name=""):
    """Return complete-block selections and the identical causal partial tail.

    Random rankings use a private CPU generator. A layer and segment length get
    the same per-query random template on every call, independent of batching,
    preceding forwards, GPU RNG, or the chosen intervention policy.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown routing policy: {policy}")
    if scores.ndim != 2 or visible.shape != scores.shape or visible.dtype != torch.bool:
        raise ValueError("Expected scores and boolean visibility [queries, blocks]")
    if block_size <= 0 or topk <= 0 or scores.shape[1] != scores.shape[0] // block_size:
        raise ValueError("Invalid complete-block geometry or top-k")
    length, blocks = scores.shape
    positions = torch.arange(length, device=scores.device)
    ends = (torch.arange(blocks, device=scores.device) + 1) * block_size - 1
    expected = ends[None, :] <= positions[:, None]
    if not torch.equal(visible, expected):
        raise ValueError("Visibility must describe causal complete blocks")
    chosen = torch.zeros_like(visible)
    if blocks:
        if policy == "learned":
            ranking = scores.detach()
        elif policy == "prefix":
            ranking = -torch.arange(blocks, device=scores.device, dtype=torch.float64).expand(length, -1)
        elif policy == "local":
            ranking = torch.arange(blocks, device=scores.device, dtype=torch.float64).expand(length, -1)
        else:
            payload = f"qsa-routing-v0\0{int(seed)}\0{layer_name}\0{length}\0{block_size}\0{blocks}".encode()
            private_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)
            generator = torch.Generator(device="cpu").manual_seed(private_seed)
            ranking = torch.rand((length, blocks), generator=generator, dtype=torch.float64).to(scores.device)
        order = ranking.masked_fill(~visible, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        chosen.scatter_(1, order[:, :min(blocks, topk)], True)
        chosen &= visible
    tail_start = ((positions + 1) // block_size) * block_size
    mask = (positions[None, :] >= tail_start[:, None]) & (positions[None, :] <= positions[:, None])
    if blocks:
        mask[:, :blocks * block_size] |= chosen.repeat_interleave(block_size, dim=-1)
    return chosen, mask


def _entropy(scores, support):
    """Empty supports have NaN, excluded from the metric's own denominator."""
    if scores.shape[1] == 0:
        return scores.new_full((scores.shape[0],), float("nan"))
    valid = support.any(-1)
    safe_scores = scores.to(_normalization_dtype(scores)).masked_fill(~support, -torch.inf)
    safe_scores = torch.where(valid[:, None], safe_scores, torch.zeros_like(safe_scores))
    log_p = safe_scores.log_softmax(-1)
    terms = torch.where(support, log_p.exp() * log_p, torch.zeros_like(log_p))
    return (-terms.sum(-1)).masked_fill(~valid, float("nan"))


class _Aggregate:
    def __init__(self):
        self.query_count = 0
        self.metrics = {}

    def add(self, features, mask):
        self.query_count += int(mask.sum())
        for name, values in features.items():
            selected = values[mask]
            selected = selected[torch.isfinite(selected)]
            if not selected.numel():
                continue
            item = self.metrics.setdefault(name, {"count": 0, "sum": 0., "sum_squares": 0.,
                                                   "min": float("inf"), "max": -float("inf")})
            item["count"] += selected.numel()
            item["sum"] += float(selected.sum())
            item["sum_squares"] += float(selected.square().sum())
            item["min"] = min(item["min"], float(selected.min()))
            item["max"] = max(item["max"], float(selected.max()))

    def summary(self):
        metrics = copy.deepcopy(self.metrics)
        for item in metrics.values():
            item["mean"] = item["sum"] / item["count"]
        # Rates are per query, not per head or block. Their explicit count is
        # the denominator, and sum the numerator; no vacuous empty-row zeros.
        return {"query_count": self.query_count, "metrics": metrics}


class RoutingDiagnostics:
    """Temporarily observe GlobalAttention and optionally intervene on routing.

    Example::

        with torch.no_grad(), RoutingDiagnostics(model, policy="learned") as d:
            result = model(...)
        report = d.summary()

    Only sparse-mode segments are observed. The supplied supervised-query mask
    is honored exactly; the LM caller excludes each segment's final query.
    This context does not change training/eval flags, parameters, or gradients.
    """

    def __init__(self, model, policy="learned", seed=20260920,
                 position_edges=(256, 512, 1024, 2048)):
        if policy not in POLICIES:
            raise ValueError(f"Unknown routing policy: {policy}")
        if any(int(x) != x or x <= 0 for x in position_edges) or list(position_edges) != sorted(set(position_edges)):
            raise ValueError("Position edges must be strictly increasing positive integers")
        self.model, self.policy, self.seed = model, policy, int(seed)
        self.position_edges = tuple(int(x) for x in position_edges)
        self.layers = [(name or "<root>", module) for name, module in model.named_modules()
                       if isinstance(module, GlobalAttention)]
        if not self.layers:
            raise ValueError("No GlobalAttention instances found")
        if any(layer.indexer is None for _, layer in self.layers):
            raise ValueError("Every observed layer must have an initialized sparse indexer")
        self._saved, self._aggregates, self._layer_info = [], {}, {}
        self._entered = False
        self.forward_counts = {"observed_sparse_segment_forwards": 0,
                               "extra_indexer_projection_passes": 0,
                               "extra_indexer_dot_product_calls": 0,
                               "extra_main_qkv_projection_passes": 0,
                               "extra_full_causal_attention_calls": 0,
                               "extra_counterfactual_policy_selections": 0,
                               "original_sparse_selection_calls": 0,
                               "additional_full_lm_forwards": 0,
                               "diagnostic_backward_calls": 0, "optimizer_updates": 0}

    def __enter__(self):
        if self._entered or self._saved:
            raise RuntimeError("RoutingDiagnostics contexts are single-use")
        if any(layer in _ACTIVE for _, layer in self.layers):
            raise RuntimeError("Layer already has active routing diagnostics")
        self._entered = True
        try:
            for name, layer in self.layers:
                self._saved.append((layer, {key: (key in layer.__dict__, layer.__dict__.get(key))
                                            for key in ("_segment", "_sparse_selection")}))
                original_segment = layer._segment

                def segment(this, x, mode, supervised_queries, _name=name, _original=original_segment):
                    if mode == "sparse":
                        with torch.no_grad():
                            self._observe(_name, this, x, supervised_queries)
                        self.forward_counts["observed_sparse_segment_forwards"] += 1
                    return _original(x, mode, supervised_queries)

                def selection(this, scores, visible, _name=name):
                    self.forward_counts["original_sparse_selection_calls"] += 1
                    return select_support(scores, visible, block_size=this.block_size,
                                          topk=this.selected_blocks, policy=self.policy,
                                          seed=self.seed, layer_name=_name)

                layer._segment = types.MethodType(segment, layer)
                layer._sparse_selection = types.MethodType(selection, layer)
                _ACTIVE[layer] = self
                indexer = layer.indexer
                self._layer_info[name] = {
                    "block_size": layer.block_size, "topk_complete_blocks": layer.selected_blocks,
                    "index_score_scale": indexer.score_scale, "index_head_dim": indexer.dim,
                    "index_heads": indexer.heads,
                    "index_q_effective_rms_gain_l2": float((1 + indexer.q_norm.weight.detach().double()).norm()),
                    "index_k_effective_rms_gain_l2": float((1 + indexer.k_norm.weight.detach().double()).norm()),
                }
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        for layer, saved in reversed(self._saved):
            for key, (was_present, value) in saved.items():
                if was_present:
                    setattr(layer, key, value)
                else:
                    delattr(layer, key)
            if _ACTIVE.get(layer) is self:
                del _ACTIVE[layer]
        self._saved.clear()

    def __exit__(self, exc_type, exc, traceback):
        self._restore()
        return False

    def _observe(self, name, layer, x, supervised_queries):
        length, indexer = x.shape[0], layer.indexer
        blocks = length // layer.block_size
        if supervised_queries.shape != (length,):
            raise ValueError("Supervised-query mask must match segment length")
        hidden = x.detach()
        index_q = indexer.q_norm(indexer.q_proj(hidden).reshape(length, indexer.heads, indexer.dim))
        raw_k = indexer.k_proj(hidden)
        pooled = raw_k[:blocks * layer.block_size].reshape(blocks, layer.block_size, indexer.dim).float().mean(1)
        index_k = indexer.k_norm(pooled.to(raw_k.dtype))
        positions = torch.arange(length, device=x.device)
        starts = torch.arange(blocks, device=x.device) * layer.block_size
        index_q = _rope(index_q, positions, indexer.rotary_dim, indexer.rope_theta)
        index_k = _rope(index_k, starts, indexer.rotary_dim, indexer.rope_theta)
        pre_relu = torch.einsum("thd,bd->thb", index_q, index_k)
        scores = pre_relu.relu().sum(1) * indexer.score_scale
        visible = (starts + layer.block_size - 1)[None, :] <= positions[:, None]
        visible_count = visible.sum(-1)
        nonempty = visible_count > 0
        effective = visible_count > layer.selected_blocks
        if not torch.isfinite(scores).all() or not torch.isfinite(pre_relu).all():
            raise FloatingPointError("Nonfinite router projections/scores in diagnostic")
        self.forward_counts["extra_indexer_projection_passes"] += 1
        self.forward_counts["extra_indexer_dot_product_calls"] += 1
        query, key, _value = layer._main_qkv(hidden)
        logits = torch.einsum("thd,shd->ths", query, key) * layer.head_dim ** -0.5
        causal = positions[None, :] <= positions[:, None]
        full_probability = logits.masked_fill(~causal[:, None, :], -torch.inf).softmax(
            -1, dtype=_normalization_dtype(logits)).mean(1)
        if not torch.isfinite(full_probability).all():
            raise FloatingPointError("Nonfinite full causal attention in diagnostic")
        self.forward_counts["extra_main_qkv_projection_passes"] += 1
        self.forward_counts["extra_full_causal_attention_calls"] += 1
        nan = scores.new_full((length,), float("nan"), dtype=torch.float64)
        features = {"visible_complete_blocks": visible_count.double(),
                    "index_query_l2_mean_heads": index_q.double().norm(dim=-1).mean(-1)}
        selections = {}
        for policy in POLICIES:
            chosen, mask = select_support(scores, visible, block_size=layer.block_size,
                                          topk=layer.selected_blocks, policy=policy,
                                          seed=self.seed, layer_name=name)
            selections[policy] = chosen
            features[f"coverage_{policy}"] = (full_probability.double() * mask).sum(-1)
            self.forward_counts["extra_counterfactual_policy_selections"] += 1
        if blocks:
            features["zero_score_query"] = ((scores.masked_fill(~visible, 0).sum(-1) == 0).double()).masked_fill(~nonempty, float("nan"))
            features["all_pre_relu_nonpositive_query"] = ((pre_relu <= 0) | ~visible[:, None, :]).all(-1).all(-1).double().masked_fill(~nonempty, float("nan"))
            features["all_pre_relu_strict_negative_query"] = ((pre_relu < 0) | ~visible[:, None, :]).all(-1).all(-1).double().masked_fill(~nonempty, float("nan"))
            score_min = scores.masked_fill(~visible, torch.inf).min(-1).values
            score_max = scores.masked_fill(~visible, -torch.inf).max(-1).values
            features["visible_score_min"] = score_min.masked_fill(~nonempty, float("nan"))
            features["visible_score_max"] = score_max.masked_fill(~nonempty, float("nan"))
            features["visible_score_span"] = (score_max - score_min).masked_fill(~nonempty, float("nan"))
            z = pre_relu.double()
            denom = (visible_count * indexer.heads).clamp_min(1)
            features["pre_relu_mean"] = (z.masked_fill(~visible[:, None, :], 0).sum((1, 2)) / denom).masked_fill(~nonempty, float("nan"))
            features["pre_relu_positive_fraction"] = (((z > 0) & visible[:, None, :]).sum((1, 2)) / denom).masked_fill(~nonempty, float("nan"))
            features["pre_relu_min"] = z.masked_fill(~visible[:, None, :], torch.inf).amin((1, 2)).masked_fill(~nonempty, float("nan"))
            features["pre_relu_max"] = z.masked_fill(~visible[:, None, :], -torch.inf).amax((1, 2)).masked_fill(~nonempty, float("nan"))
            k_norm = index_k.double().norm(dim=-1)
            features["index_visible_key_l2_mean"] = (visible * k_norm[None, :]).sum(-1).div(visible_count.clamp_min(1)).masked_fill(~nonempty, float("nan"))
            selected_count = selections["learned"].sum(-1)
            for policy in ("prefix", "local", "random"):
                overlap = (selections["learned"] & selections[policy]).sum(-1) / selected_count.clamp_min(1)
                features[f"learned_{policy}_complete_block_overlap"] = overlap.masked_fill(~nonempty, float("nan"))
        else:
            for key_name in ("zero_score_query", "all_pre_relu_nonpositive_query", "all_pre_relu_strict_negative_query",
                             "visible_score_min", "visible_score_max", "visible_score_span", "pre_relu_mean",
                             "pre_relu_positive_fraction", "pre_relu_min", "pre_relu_max", "index_visible_key_l2_mean",
                             "learned_prefix_complete_block_overlap", "learned_local_complete_block_overlap",
                             "learned_random_complete_block_overlap"):
                features[key_name] = nan
        features["visible_score_entropy_nats"] = _entropy(scores, visible)
        features["learned_selected_score_entropy_nats"] = _entropy(scores, selections["learned"])
        # One small [queries, scalar metrics] host transfer; never retain an
        # attention matrix, a hidden state, or per-query values in this object.
        names = list(features)
        values_cpu = torch.stack([features[k].double() for k in names], -1).cpu()
        features_cpu = {k: values_cpu[:, i] for i, k in enumerate(names)}
        positions_cpu = torch.arange(1, length + 1)
        all_mask = torch.ones(length, dtype=torch.bool)
        buckets = {"all": all_mask}
        lower = 0
        for upper in self.position_edges:
            buckets[f"{lower + 1}-{upper}"] = (positions_cpu > lower) & (positions_cpu <= upper)
            lower = upper
        buckets[f"{lower + 1}+"] = positions_cpu > lower
        strata = {"all": all_mask, "nonempty_visible": nonempty.cpu(), "effective_topk": effective.cpu()}
        query_sets = {"all_queries": all_mask, "supervised_queries": supervised_queries.detach().bool().cpu()}
        layer_aggs = self._aggregates.setdefault(name, {})
        for query_name, query_mask in query_sets.items():
            for bucket_name, bucket_mask in buckets.items():
                for stratum_name, stratum_mask in strata.items():
                    agg = layer_aggs.setdefault((query_name, bucket_name, stratum_name), _Aggregate())
                    agg.add(features_cpu, query_mask & bucket_mask & stratum_mask)

    def summary(self):
        layers = copy.deepcopy(self._layer_info)
        for name, aggregates in self._aggregates.items():
            for (query_name, bucket_name, stratum_name), aggregate in aggregates.items():
                layers[name].setdefault(query_name, {}).setdefault(bucket_name, {})[stratum_name] = aggregate.summary()
        return {"schema": "babylm-routing-diagnostics-v0", "intervention_policy": self.policy,
                "random_seed": self.seed, "position_edges": list(self.position_edges),
                "position_basis": "one-based position within each contiguous segment",
                "random_template": "private CPU RNG; seed+layer+segment length+block geometry; equal-length segments share templates; no forward-call index",
                "coverage_definition": "sum of retained token probability under full causal main attention, averaged over query heads on the same layer input; softmax before value-dtype rounding",
                "denominators": {"nonempty_visible": "complete visible blocks > 0",
                                 "effective_topk": "complete visible blocks > topk",
                                 "supervised_queries": "exact mask supplied to frozen _segment by caller; LM excludes final query",
                                 "rates": "metric sum / metric count; query-level, never empty-support rows",
                                 "entropy": "nats at unchanged index score scale; empty support excluded"},
                "interpretation": "same-hidden comparisons are within one observed forward; interventions can change later-layer hidden states across separate forwards",
                "forward_counts": copy.deepcopy(self.forward_counts), "layers": layers}
