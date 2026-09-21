"""Token-weighted development NLL on the shared, fixed window reader."""
from __future__ import annotations
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from scripts.prepare_babylm_windows_v0 import iter_windows


# Fixed before any scientific D/E scores. These are query-history lengths,
# including the current query token, not document positions or GDN resets.
POSITION_BINS = ((1, 256), (257, 512), (513, 1024), (1025, 2048))


def _loss_parts(token_losses):
    values = token_losses.detach().double().cpu()
    return {f"{lo}-{hi}": {"loss_tokens": int(values[lo - 1:hi].numel()),
                            "nll_sum": float(values[lo - 1:hi].sum())}
            for lo, hi in POSITION_BINS}


def _aggregate_parts(rows):
    result = {}
    for lo, hi in POSITION_BINS:
        name = f"{lo}-{hi}"
        count = sum(row[name]["loss_tokens"] for row in rows)
        loss = math.fsum(row[name]["nll_sum"] for row in rows)
        result[name] = {"loss_tokens": count, "nll_sum": loss,
                        "nll": loss / count if count else None}
    return result


def _field(output, name):
    return output[name] if isinstance(output, dict) else getattr(output, name)


def _bucket(source):
    return {"source": source, "windows": 0, "word_exposures": 0, "input_tokens": 0,
            "loss_tokens": 0, "forward_calls": 0, "zero_target_windows": 0,
            "_nll_sums": [], "_attention_counts": Counter()}


def _finish(bucket):
    result = {k: v for k, v in bucket.items() if not k.startswith("_")}
    result["nll_sum"] = math.fsum(bucket["_nll_sums"])
    result["nll"] = result["nll_sum"] / result["loss_tokens"] if result["loss_tokens"] else None
    result["attention_counts"] = dict(bucket["_attention_counts"])
    dense = result["attention_counts"].get("logical_dense_causal_pairs", 0)
    kept = result["attention_counts"].get("logical_kept_pairs", 0)
    result["logical_retained_fraction"] = kept / dense if dense else None
    return result


def evaluate_windows(model, manifest_path, window_indices, device="cpu", max_windows=None,
                     position_diagnostics=False):
    """Evaluate one independent window per forward and restore all mode flags.

    Model must already reside on ``device``; this function does not move model
    parameters or update them. NLL uses ``lm_loss``, NEVER the combined auxiliary
    objective. A zero-target window is forwarded and counted but contributes no
    NLL, even if a generic model returns an undefined loss for that window.
    ``None`` indices means the entire manifest. This API does not mark a run as
    scientific vs engineering: the caller must record its actual purpose.
    Optional position diagnostics reuse the SAME forward's logits. They add
    a loss reduction, not model calls, and require at most 2048 input tokens.
    """
    path = Path(manifest_path)
    blob = path.read_bytes()
    manifest = json.loads(blob)
    if type(position_diagnostics) is not bool:
        raise ValueError("position_diagnostics must be a bool")
    if max_windows is not None and (isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows < 0):
        raise ValueError("max_windows must be a nonnegative integer or None")
    selected = list(range(manifest["total_windows"])) if window_indices is None else list(window_indices)
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < manifest["total_windows"] for i in selected):
        raise ValueError("window_indices must contain valid integer indices")
    if len(selected) != len(set(selected)):
        raise ValueError("Repeated dev window indices are not allowed")
    if max_windows is not None:
        selected = selected[:max_windows]
    sources = {row["source_index"]: row["source"] for row in manifest["source_summaries"]}
    buckets = {sid: _bucket(name) for sid, name in sources.items()}
    total = _bucket("all")
    diagnostic_rows = []
    modes = [(module, module.training) for module in model.modules()]
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    numeric_stats = ["valid_tokens", "aux_loss_query_count", "segments", "logical_dense_causal_pairs",
                     "logical_kept_pairs", "allocated_main_score_elements", "indexer_score_elements",
                     "aux_nonempty_support_query_count", "indexer_zero_score_visible_query_count"]
    try:
        model.eval()
        with torch.no_grad():
            for item in iter_windows(path, selected):
                if not item["single_segment"] or not item["reset_model_state_before"]:
                    raise ValueError("Evaluation expects independent single-segment windows")
                # torch.tensor intentionally copies the readonly memmap view.
                inputs = torch.tensor(item["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
                if inputs.shape[1] != item["input_tokens"] or item["loss_tokens"] != max(0, inputs.shape[1] - 1):
                    raise ValueError("Window accounting does not match next-token loss policy")
                if position_diagnostics and inputs.shape[1] > POSITION_BINS[-1][1]:
                    raise ValueError("Position diagnostic protocol is fixed to at most 2048 input tokens")
                output = model(inputs, aux_weight=0.0)
                count = int(_field(output, "token_loss_count"))
                if count != item["loss_tokens"]:
                    raise ValueError(f"Model/ledger loss-token mismatch: {count} vs {item['loss_tokens']}")
                if count:
                    loss = float(_field(output, "lm_loss").detach().double().item())
                    if not math.isfinite(loss):
                        raise ValueError("Nonfinite LM loss with nonempty targets")
                    nll_sum = loss * count
                else:
                    nll_sum = 0.0
                if position_diagnostics:
                    logits = _field(output, "logits")
                    if logits.ndim != 3 or logits.shape[:2] != inputs.shape:
                        raise ValueError("Diagnostic logits must align with input_ids [1,T,V]")
                    per_token = (F.cross_entropy(logits[0, :-1], inputs[0, 1:], reduction="none")
                                 if count else logits.new_empty((0,)))
                    if not bool(torch.isfinite(per_token).all()):
                        raise ValueError("Nonfinite position diagnostic loss")
                    exact_sum = float(per_token.detach().double().sum().item())
                    # Reductions can differ at rounding precision, but an
                    # unrelated model-provided LM loss must not be accepted.
                    if not math.isclose(exact_sum, nll_sum, rel_tol=1e-5, abs_tol=1e-5):
                        raise ValueError("Diagnostic logits/LM-loss mismatch")
                    length_bin = next(f"{lo}-{hi}" for lo, hi in POSITION_BINS if lo <= inputs.shape[1] <= hi)
                    diagnostic_rows.append({"window_index": int(item["window_index"]),
                        "source": sources[item["source_index"]],
                        "segment_index_in_source": item.get("segment_index_in_source"),
                        "source_token_start": item.get("source_token_start"),
                        "source_token_end": item.get("source_token_end"),
                        "input_tokens": int(inputs.shape[1]), "loss_tokens": count,
                        "window_length_bin": length_bin, "nll_sum": exact_sum,
                        "nll": exact_sum / count if count else None,
                        "query_history_bins": _loss_parts(per_token)})
                stats = _field(output, "attention_stats")
                counts = Counter()
                for stat in stats:
                    for key in numeric_stats:
                        if key in stat:
                            counts[key] += int(stat[key])
                for bucket in [total, buckets[item["source_index"]]]:
                    bucket["windows"] += 1
                    bucket["word_exposures"] += int(item["word_exposures"])
                    bucket["input_tokens"] += int(item["input_tokens"])
                    bucket["loss_tokens"] += count
                    bucket["forward_calls"] += 1
                    bucket["zero_target_windows"] += int(count == 0)
                    bucket["_nll_sums"].append(nll_sum)
                    bucket["_attention_counts"].update(counts)
    finally:
        # Preserve unusual mixed train/eval submodule modes as well as root mode.
        for module, mode in modes:
            module.training = mode
    result = {"status": "nll_evaluation_complete", "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - begin,
            "manifest_path": str(path.resolve()), "manifest_sha256": hashlib.sha256(blob).hexdigest(),
            "window_indices": selected, "device": str(device),
            "aggregation": "sum(window_lm_loss * effective_next_token_targets) / sum(targets)",
            "nll_units": "natural_log_per_loss_token", "auxiliary_loss_included_in_nll": False,
            "total": _finish(total), "per_source": {sources[sid]: _finish(bucket) for sid, bucket in buckets.items()},
            "model_training_mode_restored": all(module.training == mode for module, mode in modes),
            "grad_enabled_during_model_forward": False, "optimizer_updates": 0,
            "attention_counts_scope": "Sum over all model global layers; reference allocations are not runtime speed measurements"}
    if position_diagnostics:
        by_length = {}
        for lo, hi in POSITION_BINS:
            name = f"{lo}-{hi}"
            rows = [row for row in diagnostic_rows if row["window_length_bin"] == name]
            count = sum(row["loss_tokens"] for row in rows)
            loss = math.fsum(row["nll_sum"] for row in rows)
            by_length[name] = {"windows": len(rows), "loss_tokens": count, "nll_sum": loss,
                               "nll": loss / count if count else None}
        result["position_diagnostics"] = {
            "protocol": "babylm-dev-position-v0", "extra_model_forwards": 0,
            "position_definition": "1-based query history length within reset window; target is the next token",
            "query_history_bins": _aggregate_parts([r["query_history_bins"] for r in diagnostic_rows]),
            "per_source_query_history_bins": {
                name: _aggregate_parts([r["query_history_bins"] for r in diagnostic_rows if r["source"] == name])
                for name in sources.values()},
            "window_length_bins": by_length, "per_window": diagnostic_rows,
            "scope": "Descriptive diagnostics, not independent token samples or proof of long-range causality; no scores used to choose bins or windows"}
    return result
