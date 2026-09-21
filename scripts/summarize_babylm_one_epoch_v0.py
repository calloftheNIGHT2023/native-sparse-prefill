#!/usr/bin/env python3
"""Audit a completed one-epoch run and summarize its final online updates.

Standard library only: this script never imports a model or performs a forward.
Training PPL is exp(LM cross entropy), never exp(the auxiliary/combined loss).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


class AuditError(ValueError):
    """The logs do not establish the frozen one-epoch endpoint."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def finite(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{name}: expected a number")
    require(math.isfinite(value), f"{name}: nonfinite value")
    return float(value)


def integer(value, name, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"{name}: expected integer >= {minimum}")
    return value


def safe_exp(value):
    try:
        result = math.exp(value)
    except OverflowError as exc:
        raise AuditError("PPL overflow; no finite result can be reported") from exc
    require(math.isfinite(result), "PPL is nonfinite")
    return result


def reject_constant(value):
    raise AuditError(f"Nonfinite JSON constant: {value}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def is_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def limits(values):
    return {"min": min(values), "max": max(values), "count": len(values)}


def control_ranges(updates):
    learning_rates, coefficients = {}, {}
    for event in updates:
        for group, rate in event["lr"].items():
            learning_rates.setdefault(group, []).append(rate)
        for group, value in event["_clip_coefficients"].items():
            coefficients.setdefault(group, []).append(value)
    return {"learning_rates": {k: limits(v) for k, v in learning_rates.items()},
            "clip_coefficients": {k: limits(v) for k, v in coefficients.items()},
            "clip_coefficient_sources": sorted({event["_clip_source"] for event in updates}),
            "gradient_clip_scopes": sorted({event.get("gradient_clip_scope", "global")
                                            for event in updates})}


def validate_learning_rate(event, protocol, words):
    """Independently recompute the frozen word-based warmup/cosine schedule."""
    base_fields = {"max_word_exposures", "warmup_word_exposures", "min_lr_ratio", "learning_rate"}
    if not base_fields.issubset(protocol):
        return False
    cap = integer(protocol["max_word_exposures"], "max_word_exposures", 1)
    warmup = integer(protocol["warmup_word_exposures"], "warmup_word_exposures")
    minimum = finite(protocol["min_lr_ratio"], "min_lr_ratio")
    require(warmup <= cap and 0 <= minimum <= 1, "Invalid frozen LR schedule")
    require(event.get("lr_word_position") == words,
            "LR word position does not match cumulative word exposures")
    if warmup and words < warmup:
        multiplier = max(0.0, float(words) / warmup)
    else:
        progress = min(1.0, max(0.0, (float(words) - warmup) / max(cap - warmup, 1)))
        multiplier = minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))
    recorded_multiplier = finite(event.get("lr_multiplier"), "lr_multiplier")
    require(math.isclose(recorded_multiplier, multiplier, rel_tol=1e-12, abs_tol=1e-15),
            "Logged LR multiplier does not match frozen schedule")
    for group, observed in event["lr"].items():
        require(group in {"backbone", "indexer"}, "Unknown optimizer group for frozen LR schedule")
        field = "indexer_learning_rate" if group == "indexer" else "learning_rate"
        peak = finite(protocol.get(field), field)
        require(peak > 0, "Nonpositive frozen peak learning rate")
        require(math.isclose(observed, peak * multiplier, rel_tol=1e-12, abs_tol=1e-15),
                f"{group} LR does not match frozen schedule")
    return True


def summarize(run_dir, tail_updates=100):
    run_dir = Path(run_dir)
    integer(tail_updates, "tail_updates", 1)
    paths = {name: run_dir / name for name in ("protocol.json", "summary.json", "events.jsonl")}
    before = {name: sha256(path) for name, path in paths.items()}
    protocol, summary = read_json(paths["protocol.json"]), read_json(paths["summary.json"])
    require(protocol.get("max_epochs") == 1, "Protocol must freeze max_epochs=1")
    require(protocol.get("expected_stop_reason") == "epoch_complete",
            "Protocol must expect epoch_complete")
    require(summary.get("status") == "epoch_complete", "Run is not epoch_complete")
    require(summary.get("scope") == protocol.get("scope") == "scientific",
            "Expected a scientific run, not engineering/preflight")
    expected = protocol.get("expected_epoch_counts")
    require(isinstance(expected, dict), "Missing expected_epoch_counts")
    count_fields = ("windows", "updates", "word_exposures", "input_tokens", "loss_tokens",
                    "forward_calls", "backward_calls")
    for key in count_fields + ("final_update_windows",):
        integer(expected.get(key), f"expected_epoch_counts.{key}", 1)
    batch = integer(protocol.get("windows_per_update"), "windows_per_update", 1)
    require(expected["updates"] == math.ceil(expected["windows"] / batch),
            "Expected update count does not match epoch window count")
    require(expected["final_update_windows"] == (expected["windows"] - 1) % batch + 1,
            "Incorrect frozen final partial-batch size")
    require(tail_updates <= expected["updates"], "Not enough updates for the requested tail")
    phash = canonical_sha(protocol)
    require(summary.get("protocol_sha256") == phash, "Summary protocol SHA mismatch")
    sources = summary.get("source_hashes")
    require(isinstance(sources, dict) and bool(sources)
            and all(is_sha(value) for value in sources.values()), "Invalid source SHA map")
    fingerprint = summary.get("data_fingerprint")
    require(isinstance(fingerprint, dict)
            and fingerprint.get("total_windows") == expected["windows"],
            "Dataset window count mismatch")
    if "train_manifest_sha256" in protocol:
        require(fingerprint.get("manifest_sha256") == protocol["train_manifest_sha256"],
                "Training manifest SHA mismatch")

    totals = {key: 0 for key in count_fields}
    seen = set()
    updates = []
    start = stop = None
    previous_id = 0
    lr_groups = None
    with paths["events.jsonl"].open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"Blank event line {line_number}")
            event = json.loads(line, parse_constant=reject_constant)
            event_id = integer(event.get("event_id"), "event_id", 1)
            require(event_id == previous_id + 1, f"Discontinuous event_id at line {line_number}")
            previous_id = event_id
            kind = event.get("type")
            require(kind not in {"resume", "run_resume", "run_error", "failure", "consumed_without_update"},
                    f"Unsupported failure/resume/skipped-window event: {kind}")
            if kind == "run_start":
                require(start is None and not updates and stop is None, "Multiple or late run_start")
                start = event
                require(event.get("protocol_sha256") == phash, "Start protocol SHA mismatch")
                require(event.get("source_hashes") == sources, "Start/summary source hashes differ")
                require(event.get("data_fingerprint") == fingerprint, "Start/summary data differ")
                require(event.get("cursor") == {"epoch": 0, "position": 0}, "Run did not start at epoch 0")
                require(all(event.get("counts", {}).get(key) == 0 for key in count_fields),
                        "Run did not start with zero scientific counts")
            elif kind == "run_stop":
                require(start is not None and stop is None, "Missing start or duplicate stop")
                require(event.get("status") == "epoch_complete", "Terminal event is not epoch_complete")
                stop = event
            elif kind == "update":
                require(start is not None and stop is None, "Update outside a single active run")
                number = len(updates) + 1
                require(number <= expected["updates"], "Extra update after the frozen endpoint")
                windows = event.get("windows")
                required_batch = (expected["final_update_windows"]
                                  if number == expected["updates"] else batch)
                require(isinstance(windows, list) and len(windows) == required_batch,
                        f"Update {number}: wrong batch size, including final partial batch")
                identifiers, update_tokens = [], 0
                for window in windows:
                    require(window.get("epoch") == 0, "Repeated/multiple epoch detected")
                    index = integer(window.get("window_index"), "window_index")
                    require(index < expected["windows"] and index not in seen,
                            f"Duplicate or out-of-range window {index}")
                    seen.add(index)
                    identifiers.append({"epoch": 0, "window_index": index})
                    for key in ("forward_started", "forward_completed", "backward_started", "backward_completed"):
                        require(window.get(key) is True, f"Incomplete {key} for window {index}")
                    for key in ("word_exposures", "input_tokens", "loss_tokens"):
                        value = integer(window.get(key), key, 1 if key != "word_exposures" else 0)
                        totals[key] += value
                    require(window["loss_tokens"] == window["input_tokens"] - 1,
                            "Window loss-token accounting is not causal next-token accounting")
                    update_tokens += window["loss_tokens"]
                    totals["windows"] += 1
                    totals["forward_calls"] += 1
                    totals["backward_calls"] += 1
                totals["updates"] += 1
                require(event.get("window_ids") == identifiers, "window_ids/windows disagreement")
                require(event.get("loss_tokens_this_update") == update_tokens, "Update loss-token mismatch")
                counts = event.get("counts", {})
                require(all(counts.get(key) == value for key, value in totals.items()),
                        f"Cumulative count mismatch at update {number}")
                require(counts.get("scientific_updates") == number and counts.get("engineering_updates") == 0,
                        "Scientific/engineering update counts disagree")
                for key, expected_value in (("forward_attempts", totals["forward_calls"]),
                                            ("backward_attempts", totals["backward_calls"]),
                                            ("forward_input_tokens", totals["input_tokens"]),
                                            ("forward_word_exposures", totals["word_exposures"]),
                                            ("skipped_zero_target_input_tokens", 0),
                                            ("skipped_zero_target_word_exposures", 0)):
                    require(counts.get(key) == expected_value, f"Inconsistent {key} at update {number}")
                cursor = ({"epoch": 1, "position": 0} if number == expected["updates"] else
                          {"epoch": 0, "position": totals["windows"]})
                require(event.get("cursor") == cursor, f"Cursor mismatch at update {number}")
                ce = finite(event.get("loss_token_weighted_ce"), "LM cross entropy")
                require(ce >= 0, "Negative LM cross entropy")
                finite(event.get("loss_token_weighted_aux"), "auxiliary loss")
                for key in ("weighted_aux_loss", "loss_token_weighted_combined_objective", "grad_norm",
                            "backbone_grad_norm", "indexer_grad_norm", "aux_weight", "aux_weight_applied"):
                    if key in event:
                        finite(event[key], key)
                rates = event.get("lr")
                require(isinstance(rates, dict) and "backbone" in rates, "Missing learning rates")
                require(lr_groups is None or set(rates) == lr_groups, "Optimizer LR groups changed")
                lr_groups = set(rates)
                for group, value in rates.items():
                    require(finite(value, f"lr.{group}") >= 0, "Negative learning rate")
                event["_lr_schedule_verified"] = validate_learning_rate(event, protocol, totals["word_exposures"])
                clip = {}
                clip_source = "separate_group_fields"
                for group in rates:
                    key = next((candidate for candidate in (f"{group}_grad_clip_coefficient",
                                                           f"{group}_clip_coefficient") if candidate in event), None)
                    if key is None:
                        require(event.get("gradient_clip_scope", "global") == "global",
                                "Missing separate-group clipping coefficient")
                        key = "grad_clip_coefficient"
                        clip_source = "legacy_global_coefficient"
                    value = finite(event.get(key), key)
                    require(0 <= value <= 1, "Clipping coefficient outside [0,1]")
                    clip[group] = value
                event["_clip_coefficients"], event["_clip_source"] = clip, clip_source
                updates.append(event)

    require(start is not None and stop is not None, "Missing explicit completed run boundaries")
    require(len(updates) == expected["updates"] and len(seen) == expected["windows"],
            "Incomplete one-epoch coverage")
    require(seen == set(range(expected["windows"])), "Not every epoch-0 window was observed exactly once")
    require(all(totals[key] == expected[key] for key in count_fields), "Frozen epoch counts not reached exactly")
    for name, record in (("summary", summary), ("run_stop", stop)):
        require(record.get("counts") == updates[-1]["counts"], f"{name}/final update counts differ")
        require(record.get("cursor") == {"epoch": 1, "position": 0}, f"{name}: incomplete epoch cursor")
    require(summary.get("eval_counts") == stop.get("eval_counts"), "Evaluation accounting differs")
    require(summary.get("last_lr") == updates[-1]["lr"], "Final learning rates disagree")
    require(before == {name: sha256(path) for name, path in paths.items()},
            "Input files changed during audit")

    tail = updates[-tail_updates:]
    ce = [row["loss_token_weighted_ce"] for row in tail]
    aux = [row["loss_token_weighted_aux"] for row in tail]
    tokens = [row["loss_tokens_this_update"] for row in tail]
    n_tokens = sum(tokens)
    weighted_ce = math.fsum(loss * count for loss, count in zip(ce, tokens)) / n_tokens
    weighted_aux = math.fsum(loss * count for loss, count in zip(aux, tokens)) / n_tokens
    mean_ce = statistics.fmean(ce)
    metrics = {"token_weighted_lm_nll": weighted_ce, "token_weighted_lm_ppl": safe_exp(weighted_ce),
               "arithmetic_mean_step_lm_nll": mean_ce,
               "arithmetic_mean_step_lm_ppl": statistics.fmean(safe_exp(value) for value in ce),
               "exp_arithmetic_mean_step_lm_nll": safe_exp(mean_ce),
               "auxiliary_loss_separate": {"token_weighted_mean": weighted_aux,
                                           "arithmetic_mean_step": statistics.fmean(aux)}}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            finite(value, key)
    require(all(math.isfinite(value) for value in metrics["auxiliary_loss_separate"].values()),
            "Auxiliary aggregation overflow")
    return {"schema_version": 1, "status": "audited_epoch_complete", "mode": summary.get("mode"),
            "scope": "online_training_loss_on_changing_model_and_training_minibatches_not_heldout_evaluation",
            "interpretation": "No held-out quality, equivalence, speed, or novelty claim follows from this statistic.",
            "primary_metric": "token_weighted_lm_nll_and_its_exponential_ppl",
            "tail": {"first_update": len(updates) - tail_updates + 1, "last_update": len(updates),
                     "updates": tail_updates, "loss_tokens": n_tokens,
                     "windows": sum(len(row["windows"]) for row in tail),
                     "input_tokens": sum(w["input_tokens"] for row in tail for w in row["windows"]),
                     "word_exposures": sum(w["word_exposures"] for row in tail for w in row["windows"]),
                     "metrics": metrics, "controls": control_ranges(tail)},
            "whole_run_controls": control_ranges(updates), "counts": summary["counts"],
            "lr_schedule_audit": {"all_update_lrs_verified_against_protocol":
                                  all(event["_lr_schedule_verified"] for event in updates),
                                  "verified_updates": sum(event["_lr_schedule_verified"] for event in updates),
                                  "method": "Independently recomputed word-based linear warmup and cosine decay; skipped only when base schedule fields are absent."},
            "eval_counts_recorded_by_run": summary.get("eval_counts"),
            "epoch_coverage": {"epoch": 0, "first_window_id": 0,
                               "last_window_id": expected["windows"] - 1,
                               "every_window_exactly_once": True,
                               "final_update_windows": len(updates[-1]["windows"])},
            "protocol_canonical_sha256": phash, "input_file_sha256": before,
            "source_hashes_recorded_by_training": sources,
            "source_hash_validation": "Start and summary agree; source files were not loaded or replayed by this audit.",
            "data_fingerprint": fingerprint,
            "analysis_counts": {"forward_calls": 0, "backward_calls": 0, "optimizer_updates": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tail-updates", default=100, type=int)
    args = parser.parse_args()
    try:
        require(not args.output.exists(), "Output already exists; use a new evidence filename")
        result = summarize(args.run_dir, args.tail_updates)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except (AuditError, OSError, json.JSONDecodeError, KeyError, TypeError, OverflowError) as exc:
        print(f"One-epoch audit rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "tail": result["tail"], "analysis_counts": result["analysis_counts"]},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
