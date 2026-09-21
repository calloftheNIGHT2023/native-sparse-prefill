"""Bounded B1 router-only optimization on one immutable extraction of train features.

This is a static-feature engineering diagnostic, not model pretraining, a resume,
or a quality/speed result. No old training implementation is modified. The caller
must first pass the frozen migration replay and run under a 1800-second external
process-group timeout. Default invocation validates the protocol without models.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from src.babylm_hybrid.attention import _QSAIndexer, _rope
from src.babylm_hybrid.config import HybridConfig
from scripts import run_babylm_checkpoint_eval_v0 as checkpoint_eval
from scripts import prepare_babylm_windows_v0 as windows

require, sha, resolve = checkpoint_eval.require, checkpoint_eval.sha, checkpoint_eval.resolve
canonical_sha = checkpoint_eval.canonical_sha
REQUIRED_SOURCES = (*checkpoint_eval.REQUIRED_SOURCES,
                    "scripts/run_babylm_router_fixed_features_v0.py")


def state_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_sha(value):
    return state_sha({"tensor": value})


def fresh_indexers(cfg, seed):
    """Match HybridLM.initialize_indexers RNG consumption without a new backbone."""
    result = nn.ModuleList()
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for kind in cfg.layer_types:
            if kind == "global":
                indexer = _QSAIndexer(cfg)
                indexer.apply(initialize)
                result.append(indexer)
    return result


def index_rows(indexer, hidden, query_indices):
    """Exact existing QSA formula on selected rows; gradients only to indexer."""
    hidden = hidden.detach()
    queries = torch.as_tensor(query_indices, device=hidden.device, dtype=torch.long)
    length, block_size = hidden.shape[0], indexer.block_size
    blocks = length // block_size
    q = indexer.q_norm(indexer.q_proj(hidden[queries]).reshape(-1, indexer.heads, indexer.dim))
    raw_keys = indexer.k_proj(hidden)
    pooled = raw_keys[:blocks * block_size].reshape(blocks, block_size, indexer.dim).float().mean(1)
    k = indexer.k_norm(pooled.to(raw_keys.dtype))
    starts = torch.arange(blocks, device=hidden.device) * block_size
    q = _rope(q, queries, indexer.rotary_dim, indexer.rope_theta)
    k = _rope(k, starts, indexer.rotary_dim, indexer.rope_theta)
    pre_relu = torch.einsum("qhd,bd->qhb", q, k)
    scores = pre_relu.relu().sum(1) * indexer.score_scale
    visible = starts[None, :] + block_size - 1 <= queries[:, None]
    return scores, visible, pre_relu


def topk_support(scores, visible, topk):
    require(scores.shape == visible.shape and bool(visible.any(-1).all()), "Empty or invalid routing rows")
    selected = torch.zeros_like(visible)
    order = scores.detach().masked_fill(~visible, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    selected.scatter_(1, order[:, :min(topk, scores.shape[1])], True)
    return selected & visible


def support_kl(scores, target, support):
    require(bool(support.any(-1).all()), "Empty KL support")
    log_q = scores.masked_fill(~support, -torch.inf).log_softmax(-1)
    log_q = torch.where(support, log_q, torch.zeros_like(log_q))
    return (target * (target.clamp_min(1e-30).log() - log_q)).sum(-1)


def teacher_rows(layer, hidden, query_indices, fixed_support):
    """Fixed-support original KL teacher, plus separate full dense diagnostics.

    Restrict and normalize EACH HEAD before averaging for the fit teacher. A
    renormalized dense head-average is generally a different training target.
    """
    require(not torch.is_grad_enabled(), "Teacher construction must be detached")
    length = hidden.shape[0]
    queries = torch.as_tensor(query_indices, device=hidden.device, dtype=torch.long)
    positions = torch.arange(length, device=hidden.device)
    q = layer.q_norm(layer.q_proj(hidden[queries]).reshape(-1, layer.query_heads, layer.head_dim))
    k = layer.k_norm(layer.k_proj(hidden).reshape(length, layer.kv_heads, layer.head_dim))
    q = _rope(q, queries, layer.rotary_dim, layer.rope_theta)
    k = _rope(k, positions, layer.rotary_dim, layer.rope_theta)
    k = k.repeat_interleave(layer.query_heads // layer.kv_heads, 1)
    logits = torch.einsum("qhd,shd->qhs", q, k) * layer.head_dim ** -0.5
    causal = positions[None, :] <= queries[:, None]
    blocks = length // layer.block_size
    tail = (positions[None, :] >= (((queries + 1) // layer.block_size) * layer.block_size)[:, None]) & causal
    selected_tokens = tail.clone()
    selected_tokens[:, :blocks * layer.block_size] |= fixed_support.repeat_interleave(layer.block_size, -1)
    require(not bool((selected_tokens & ~causal).any()), "Future token in fixed teacher support")
    sparse_probability = logits.masked_fill(~selected_tokens[:, None, :], -torch.inf).softmax(-1).mean(1)
    dense_probability = logits.masked_fill(~causal[:, None, :], -torch.inf).softmax(-1).mean(1)
    sparse_max = sparse_probability[:, :blocks * layer.block_size].reshape(-1, blocks, layer.block_size).amax(-1)
    sparse_max *= fixed_support
    target = sparse_max / sparse_max.sum(-1, keepdim=True)
    dense_blocks = dense_probability[:, :blocks * layer.block_size].reshape(-1, blocks, layer.block_size)
    dense_max = dense_blocks.amax(-1)
    dense_mass = dense_blocks.sum(-1)
    visible = (torch.arange(blocks, device=hidden.device)[None, :] + 1) * layer.block_size - 1 <= queries[:, None]
    dense_max *= visible
    dense_mass *= visible
    dense_target = dense_max / dense_max.sum(-1, keepdim=True)
    tail_mass = (dense_probability * tail).sum(-1)
    for value in (target, dense_target, dense_mass, tail_mass):
        require(bool(torch.isfinite(value).all()), "Nonfinite teacher")
    require(bool(((target.sum(-1) - 1).abs() <= 1e-6).all()), "Fit teacher is not normalized")
    require(bool(((dense_target.sum(-1) - 1).abs() <= 1e-6).all()), "Dense diagnostic teacher is not normalized")
    return {"target": target.detach(), "dense_target": dense_target.detach(),
            "dense_token_probability": dense_probability.detach(),
            "dense_mass": dense_mass.detach(), "tail_mass": tail_mass.detach()}


def row_metrics(scores, visible, pre_relu, feature, topk, score_scale=1.0):
    selected = topk_support(scores, visible, topk)
    oracle_mass = topk_support(feature["dense_mass"], visible, topk)
    oracle_target = topk_support(feature["dense_target"], visible, topk)
    fixed = feature["fixed_support"]
    positions = torch.arange(scores.shape[1], device=scores.device).expand_as(scores)
    local = topk_support(positions.float(), visible, topk)
    prefix = topk_support(-positions.float(), visible, topk)
    mass, tail = feature["dense_mass"], feature["tail_mass"]
    def coverage(mask):
        return (mass * mask).sum(-1) + tail
    uniform_support = visible.float() / visible.sum(-1, keepdim=True)
    target = feature["dense_target"]
    random_expected = mass.sum(-1) * selected.sum(-1) / visible.sum(-1) + tail
    nonpositive = ((pre_relu <= 0) | ~visible[:, None, :]).all(-1).all(-1)
    fixed_kl = support_kl(scores, feature["target"], fixed)
    def entropy(logits, support):
        log_p = logits.masked_fill(~support, -torch.inf).log_softmax(-1)
        return -torch.where(support, log_p.exp() * log_p, torch.zeros_like(log_p)).sum(-1)
    def span(support):
        return scores.masked_fill(~support, -torch.inf).amax(-1) - scores.masked_fill(~support, torch.inf).amin(-1)
    result = {
        "fixed_support_kl": fixed_kl,
        "fixed_support_kl_common_score_scale_1": support_kl(scores / score_scale, feature["target"], fixed),
        "full_support_kl_diagnostic": support_kl(scores, target, visible),
        "visible_score_span": span(visible), "fixed_support_score_span": span(fixed),
        "visible_student_entropy": entropy(scores, visible), "fixed_support_student_entropy": entropy(scores, fixed),
        "fixed_teacher_entropy": -(feature["target"] * feature["target"].clamp_min(1e-30).log()).sum(-1),
        "full_teacher_entropy": -(target * target.clamp_min(1e-30).log()).sum(-1),
        "zero_score_rate": ((scores * visible).sum(-1) == 0).float(),
        "all_pre_relu_nonpositive_rate": nonpositive.float(),
        "dynamic_topk_dense_mass_coverage": coverage(selected),
        "fixed_mask_dense_mass_coverage_control": coverage(fixed),
        "oracle_mass_topk_coverage": coverage(oracle_mass),
        "uniform_random_expected_coverage": random_expected,
        "oracle_minus_random_coverage": coverage(oracle_mass) - random_expected,
        "dynamic_topk_recall_target_oracle": (selected & oracle_target).sum(-1) / selected.sum(-1),
        "dynamic_vs_initial_topk_overlap": (selected & fixed).sum(-1) / selected.sum(-1),
        "prefix_coverage": coverage(prefix), "local_coverage": coverage(local),
        "target_kl_from_uniform": (target * (target.clamp_min(1e-30).log() - uniform_support.clamp_min(1e-30).log())).sum(-1),
        "fixed_target_kl_from_uniform": (feature["target"] * (feature["target"].clamp_min(1e-30).log() + fixed.sum(-1, keepdim=True).float().log())).sum(-1),
    }
    require(all(bool(torch.isfinite(value).all()) for value in result.values()), "Nonfinite metric")
    return result


def validate(protocol):
    p = protocol
    require(p.get("schema_version") == 1 and p.get("scope") == "engineering_router_fixed_features", "Explicit B1 engineering scope required")
    require(p.get("mode") == "sparse" and p.get("dtype") == "float32", "Frozen sparse FP32 checkpoint required")
    require(p.get("device") in ("cpu", "cuda"), "Explicit device required")
    require(0 < p["max_wall_seconds"] <= p["external_hard_timeout_seconds"] <= 1800, "B1 is capped at 30 minutes with a consistent hard bound")
    require(0 < p["hourly_rate_usd"] <= 2 and p["external_hard_timeout_seconds"] / 3600 * p["hourly_rate_usd"] <= p["stage_cost_cap_usd"] <= 5, "Inconsistent B1 cost cap")
    require(p["indexer_updates_per_arm"] == 200 and p["evaluate_every"] == 25, "Frozen B1 is 200 updates with evaluations 0,25,...,200")
    require(p["optimizer"] == {"betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1, "clip_norm": 1.0}, "Optimizer changed")
    cfg = HybridConfig(**p["model_config"])
    require(cfg.index_score_scale == 1 and cfg.index_head_dim == 128, "Frozen common reference scale/head dimension differs")
    arms = p["arms"]
    require(len(arms) == 4 and len({arm["name"] for arm in arms}) == 4, "Exactly four uniquely named arms required")
    require(all(arm["name"].replace("_", "").replace("-", "").isalnum() for arm in arms), "Unsafe arm filename")
    # Python pow(-.5) and reciprocal sqrt differ by one float64 ULP here.
    # Preserve each exact frozen JSON value, but validate its mathematical arm.
    for lr in (1e-3, 1e-4):
        scales = sorted(float(arm["score_scale"]) for arm in arms if arm["learning_rate"] == lr)
        require(len(scales) == 2 and scales[1] == 1.0 and math.isclose(scales[0], 1 / math.sqrt(128), rel_tol=0, abs_tol=1e-16), "Frozen LR/scale 2x2 differs")
    samples = p["sampled_windows"]
    require(len(samples) == 24 and len({s["window_index"] for s in samples}) == 24, "Exactly 24 unique training windows required")
    for sample in samples:
        require(sample["split"] in ("fit", "holdout"), "Invalid router split")
        query = sample["query_indices"]
        require(0 < len(query) <= 32 and query == sorted(set(query)) and all(type(i) is int and 259 <= i <= 2046 for i in query), "Invalid sampled queries")
    require(sum(s["split"] == "fit" for s in samples) == 12, "Exactly 12 fit and 12 holdout windows required")
    fit_ids = [s["window_index"] for s in samples if s["split"] == "fit"]
    require(sorted(p["fit_window_order"]) == sorted(fit_ids), "Fixed cycle must visit every fit window exactly once")
    require(resolve(p["train_manifest"]).parent == ROOT / "data/babylm-2026-windows-v0", "Only the frozen training ledger is allowed")
    require(all(name in p["expected_source_hashes"] for name in REQUIRED_SOURCES), "Missing source pin")
    for name, digest in p["expected_source_hashes"].items():
        require(sha(resolve(name)) == digest, "Source SHA differs: " + name)
    require(p["max_feature_cache_bytes"] <= 2**30, "Cache limit must be at most 1 GiB")
    return cfg


def append(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def evaluate(indexers, cache, cfg, counts, check_wall):
    accumulated = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for window in cache:
            check_wall()
            for layer_index, feature in enumerate(window["features"]):
                scores, visible, pre = index_rows(indexers[layer_index], feature["hidden"], window["query_indices"])
                counts["indexer_metric_forward_calls"] += 1
                metrics = row_metrics(scores, visible, pre, feature, cfg.selected_complete_blocks, indexers[layer_index].score_scale)
                key = (window["split"], layer_index)
                for name, value in metrics.items():
                    accumulated[key][name].extend(value.detach().double().cpu().tolist())
    result = {"fit": {}, "holdout": {}}
    for (split, layer_index), metrics in accumulated.items():
        result[split][str(layer_index)] = {
            "queries": len(next(iter(metrics.values()))),
            "metrics": {name: {"mean": math.fsum(values) / len(values), "min": min(values), "max": max(values)} for name, values in metrics.items()}}
    return result


def extract(model, initial_indexers, protocol, cfg, counts, check_wall, device):
    global_layers = [layer.mixer for layer in model.layers if layer.kind == "global"]
    cache, captured, handles = [], {}, []
    for layer_index, layer in enumerate(global_layers):
        def capture(module, args, i=layer_index):
            require(i not in captured, "A global layer was evaluated twice during feature extraction")
            captured[i] = args[0][0].detach().clone()
        handles.append(layer.register_forward_pre_hook(capture))
    manifest_path = resolve(protocol["train_manifest"])
    samples = {sample["window_index"]: sample for sample in protocol["sampled_windows"]}
    try:
        with torch.no_grad():
            for item in windows.iter_windows(manifest_path, list(samples), verify_hashes=False):
                check_wall()
                sample = samples[item["window_index"]]
                require(item["single_segment"] and item["reset_model_state_before"] and item["input_tokens"] >= 1024, "Invalid extraction window")
                queries = sample["query_indices"]
                require(max(queries) < item["input_tokens"] - 1 and all((q + 1) // cfg.block_size > cfg.selected_complete_blocks for q in queries), "Queries must be supervised and genuinely sparse")
                captured.clear()
                inputs = torch.as_tensor(item["input_ids"].copy(), device=device, dtype=torch.long)[None, :]
                counts["backbone_forward_attempts"] += 1
                prediction = model(inputs, aux_weight=0)
                counts["backbone_forward_calls"] += 1
                counts["feature_input_tokens"] += item["input_tokens"]
                counts["feature_word_exposures"] += item["word_exposures"]
                require(prediction.token_loss_count == item["loss_tokens"] and bool(torch.isfinite(prediction.lm_loss)), "Invalid extraction forward")
                require(len(captured) == len(global_layers), "Missing global hidden features")
                features = []
                for layer_index, layer in enumerate(global_layers):
                    hidden = captured[layer_index]
                    score, visible, _pre = index_rows(initial_indexers[layer_index], hidden, queries)
                    counts["initial_support_indexer_forward_calls"] += 1
                    fixed = topk_support(score, visible, cfg.selected_complete_blocks)
                    require(bool((fixed.sum(-1) == cfg.selected_complete_blocks).all()), "Fixed support budget differs")
                    teacher = teacher_rows(layer, hidden, queries, fixed)
                    counts["teacher_qk_projection_calls"] += 1
                    counts["teacher_fixed_support_attention_rows"] += len(queries)
                    counts["teacher_dense_attention_rows"] += len(queries)
                    features.append({"hidden": hidden, "fixed_support": fixed, "visible": visible, **teacher})
                cache.append({"window_index": item["window_index"], "source_index": item["source_index"],
                              "segment_index_in_source": item["segment_index_in_source"],
                              "split": sample["split"], "query_indices": queries, "features": features})
                del prediction, inputs
    finally:
        for handle in handles:
            handle.remove()
    return cache


def cpu_cache(cache):
    return [{**window, "features": [{name: value.detach().cpu() for name, value in feature.items()}
                                     for feature in window["features"]]} for window in cache]


def run(protocol):
    cfg = validate(protocol)
    require(protocol.get("launch_allowed") is True, "B1 launch not frozen")
    output = resolve(protocol["output_dir"])
    require(output.is_relative_to(ROOT / "results"), "Output must stay under results")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_eval.write_json(output / "protocol.json", protocol)
    begin = time.monotonic()
    counts = {key: 0 for key in ("backbone_forward_attempts", "backbone_forward_calls", "backbone_backward_calls",
             "backbone_optimizer_updates", "feature_input_tokens", "feature_word_exposures", "initial_support_indexer_forward_calls",
             "teacher_qk_projection_calls", "teacher_fixed_support_attention_rows", "teacher_dense_attention_rows",
             "indexer_fit_forward_calls", "indexer_metric_forward_calls", "indexer_backward_calls", "indexer_optimizer_updates")}
    summary = {"status": "running", "scope": protocol["scope"], "counts": counts,
               "started_utc": datetime.now(timezone.utc).isoformat(), "arms": [],
               "claims": "Static router-only diagnostic; no new pretraining, LM quality, causal full-model or speed conclusion",
               "holdout_scope": "Training-corpus windows held out from router fitting only; backbone may have seen them",
               "fit_teacher": "Original selected-support, per-head attention normalization; not dense-teacher fitting",
               "schedule_scope": "Constant peak-LR pressure diagnostic, not the original warmup/cosine schedule",
               "automatic_monitoring_resumed": False}
    def check_wall():
        require(time.monotonic() - begin < protocol["max_wall_seconds"], "B1 wall budget exhausted; no retry/resume")
    def stop(signum, frame):
        raise InterruptedError("B1 terminated: " + str(signum))
    previous_handler = signal.signal(signal.SIGTERM, stop)
    try:
        torch.set_num_threads(protocol.get("torch_num_threads", 1))
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        actual_sources, dependency = checkpoint_eval.verify_sources(protocol)
        manifest, fingerprint = checkpoint_eval.verify_manifest(resolve(protocol["train_manifest"]), protocol["train_manifest_sha256"])
        summary["data_fingerprint"] = fingerprint
        device = torch.device(protocol["device"])
        model, provenance = checkpoint_eval.load_checkpoint_model(protocol, device, dependency["sha256"])
        summary["checkpoint"] = provenance
        require(model.mode == "sparse" and not any(p.requires_grad for p in model.parameters()), "Backbone must remain frozen sparse")
        model_hash = state_sha(model.state_dict())
        initial = fresh_indexers(cfg, protocol["indexer_seed"])
        initial_hash = state_sha(initial.state_dict())
        summary["common_initial_indexer_state_sha256"] = initial_hash
        summary["initial_parameter_hashes"] = {name: tensor_sha(value) for name, value in initial.state_dict().items()}
        initial.to(device)
        check_wall()
        cache = extract(model, initial, protocol, cfg, counts, check_wall, device)
        source_splits = defaultdict(int)
        for window in cache:
            source_splits[(window["source_index"], window["split"])] += 1
        require(len(source_splits) == 12 and set(source_splits.values()) == {2}, "B1 requires two fit/two holdout windows from each of six sources")
        require(state_sha(model.state_dict()) == model_hash and all(p.grad is None for p in model.parameters()), "Extraction mutated backbone or created gradients")
        summary["backbone_before_after_state_sha256"] = model_hash
        summary["backbone_unchanged"] = True
        del model, initial
        cpu = cpu_cache(cache)
        cache_bytes = sum(value.numel() * value.element_size() for w in cpu for f in w["features"] for value in f.values())
        require(cache_bytes <= protocol["max_feature_cache_bytes"], "Feature cache exceeds frozen bound")
        cache_path = output / "fixed-features.pt"
        with cache_path.open("xb") as stream:
            torch.save({"windows": cpu, "common_initial_indexer_state_sha256": initial_hash,
                        "initial_parameter_hashes": summary["initial_parameter_hashes"],
                        "protocol_sha256": canonical_sha(protocol), "checkpoint_sha256": protocol["checkpoint_sha256"]}, stream)
        summary["feature_cache"] = {"path": str(cache_path), "sha256": sha(cache_path), "tensor_bytes": cache_bytes,
            "windows": len(cpu), "fixed_support_sha256": canonical_sha([tensor_sha(f["fixed_support"]) for w in cpu for f in w["features"]]),
            "target_sha256": canonical_sha([tensor_sha(f["target"]) for w in cpu for f in w["features"]])}
        del cpu
        checkpoint_eval.write_json(output / "extraction-audit.json", {key: summary[key] for key in ("backbone_before_after_state_sha256", "backbone_unchanged", "common_initial_indexer_state_sha256", "feature_cache")})
        fit = {w["window_index"]: w for w in cache if w["split"] == "fit"}
        with (output / "updates.jsonl").open("x", encoding="utf-8") as update_log, (output / "metrics.jsonl").open("x", encoding="utf-8") as metric_log:
            for arm in protocol["arms"]:
                check_wall()
                indexers = fresh_indexers(cfg, protocol["indexer_seed"])
                require(state_sha(indexers.state_dict()) == initial_hash, "Arms do not share identical initial weights")
                for indexer in indexers:
                    indexer.score_scale = arm["score_scale"]
                indexers.to(device)
                optimizer = torch.optim.AdamW(indexers.parameters(), lr=arm["learning_rate"],
                    betas=tuple(protocol["optimizer"]["betas"]), eps=protocol["optimizer"]["eps"], weight_decay=protocol["optimizer"]["weight_decay"])
                arm_record = {**arm, "initial_state_sha256": initial_hash, "updates_completed": 0, "status": "running"}
                summary["arms"].append(arm_record)
                metrics = evaluate(indexers, cache, cfg, counts, check_wall)
                append(metric_log, {"arm": arm["name"], "update": 0, "metrics": metrics})
                arm_record["initial_metrics"] = metrics
                for step in range(1, protocol["indexer_updates_per_arm"] + 1):
                    check_wall()
                    window = fit[protocol["fit_window_order"][(step - 1) % len(fit)]]
                    optimizer.zero_grad(set_to_none=True)
                    losses = []
                    for indexer, feature in zip(indexers, window["features"]):
                        scores, visible, _pre = index_rows(indexer, feature["hidden"], window["query_indices"])
                        counts["indexer_fit_forward_calls"] += 1
                        require(torch.equal(visible, feature["visible"]), "Fixed feature visibility changed")
                        losses.append(support_kl(scores, feature["target"], feature["fixed_support"]).mean())
                    loss = torch.stack(losses).mean()
                    require(bool(torch.isfinite(loss)), "Nonfinite fixed-support fit loss")
                    loss.backward()
                    counts["indexer_backward_calls"] += 1
                    params = list(indexers.parameters())
                    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params), "Missing or nonfinite indexer gradients")
                    norm = torch.nn.utils.clip_grad_norm_(params, protocol["optimizer"]["clip_norm"])
                    require(bool(torch.isfinite(norm)), "Nonfinite global indexer gradient norm")
                    before = [p.detach().clone() for p in params]
                    parameter_norm = math.sqrt(sum(float(p.double().square().sum()) for p in before))
                    optimizer.step()
                    counts["indexer_optimizer_updates"] += 1
                    delta_norm = math.sqrt(sum(float((p.detach() - old).double().square().sum()) for p, old in zip(params, before)))
                    require(all(bool(torch.isfinite(p).all()) for p in params), "Nonfinite indexer state")
                    arm_record["updates_completed"] = step
                    append(update_log, {"arm": arm["name"], "update": step, "window_index": window["window_index"],
                        "sampled_queries_per_layer": len(window["query_indices"]), "fixed_support_kl": float(loss.detach()),
                        "layer_fixed_support_kl": [float(value.detach()) for value in losses],
                        "gradient_norm_before_clip": float(norm), "gradient_norm_after_clip": min(float(norm), protocol["optimizer"]["clip_norm"]),
                        "parameter_l2_before": parameter_norm, "parameter_delta_l2": delta_norm,
                        "step_to_parameter_norm_ratio": delta_norm / max(parameter_norm, 1e-30), "lr": arm["learning_rate"],
                        "score_scale": arm["score_scale"], "backbone_updates": 0})
                    if step % protocol["evaluate_every"] == 0:
                        metrics = evaluate(indexers, cache, cfg, counts, check_wall)
                        append(metric_log, {"arm": arm["name"], "update": step, "metrics": metrics})
                arm_record.update(status="complete_200_router_updates", final_metrics=metrics, final_state_sha256=state_sha(indexers.state_dict()))
                with (output / (arm["name"] + "-indexers.pt")).open("xb") as stream:
                    torch.save({"indexers": {n: v.detach().cpu() for n, v in indexers.state_dict().items()},
                                "arm": arm, "updates": 200, "protocol_sha256": canonical_sha(protocol)}, stream)
                del optimizer, indexers
        require(counts["indexer_optimizer_updates"] == 800 and counts["backbone_optimizer_updates"] == 0, "Unexpected update accounting")
        require(checkpoint_eval.verify_sources(protocol)[0] == actual_sources, "Sources changed during B1")
        summary["status"] = "complete_four_arm_router_diagnostic"
    except BaseException as error:
        summary.update(status="failed_or_incomplete", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if summary["arms"] and summary["arms"][-1]["status"] == "running":
            summary["arms"][-1]["status"] = "failed_or_incomplete"
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        summary.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_wall_seconds=time.monotonic() - begin)
        summary["estimated_worker_cost_usd"] = summary["elapsed_wall_seconds"] / 3600 * protocol["hourly_rate_usd"]
        summary["cost_scope"] = "Worker wall estimate, not invoice; no prelaunch idle or separate replay fees included"
        checkpoint_eval.write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(sha(args.protocol) == args.protocol_sha256, "Frozen B1 protocol SHA differs")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate(protocol)
    if not args.execute:
        print(json.dumps({"status": "validated_no_model_calls", "windows": 24, "router_updates": 800,
                          "backbone_updates": 0, "max_wall_seconds": protocol["max_wall_seconds"]}))
        return 0
    result = run(protocol)
    print(json.dumps({key: result[key] for key in ("status", "counts", "elapsed_wall_seconds")}))
    return 0 if result["status"] == "complete_four_arm_router_diagnostic" else 2


if __name__ == "__main__":
    raise SystemExit(main())
