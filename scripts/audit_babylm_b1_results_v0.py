"""Independent local B1 evidence audit; no model construction or forward calls.

Reads JSON/JSONL plus tensor-only feature/indexer artifacts on CPU. Does not
connect remotely, load a full LM, alter evidence, or choose a best checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def state_sha(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def tensor_sha(value):
    return state_sha({"tensor": value})


def fixed_final_screen(protocol, measurements):
    """Use only frozen update200 holdout metrics; no test-set/seed inference."""
    arms = protocol["arms"]
    reference = next(a["name"] for a in arms if a["learning_rate"] == 1e-3 and a["score_scale"] == 1.)
    base_initial = measurements[(reference, 0)]["holdout"]
    base_final = measurements[(reference, 200)]["holdout"]
    layers = sorted(base_final)
    dead_key = "all_pre_relu_nonpositive_rate"
    coverage_key = "dynamic_topk_dense_mass_coverage"
    kl_key = "fixed_support_kl"

    def metric(group, layer, key):
        return group[layer]["metrics"][key]["mean"]

    collapsed = []
    for layer in layers:
        initial = metric(base_initial, layer, dead_key)
        final = metric(base_final, layer, dead_key)
        if final - initial >= .10 or (final >= .90 and final > initial):
            collapsed.append(layer)
    candidates = []
    for arm in arms:
        if arm["name"] == reference:
            continue
        group = measurements[(arm["name"], 200)]["holdout"]
        deltas = {layer: {
            "nonpositive_rate": metric(group, layer, dead_key) - metric(base_final, layer, dead_key),
            "coverage": metric(group, layer, coverage_key) - metric(base_final, layer, coverage_key),
            "fixed_support_kl": metric(group, layer, kl_key) - metric(base_final, layer, kl_key),
        } for layer in layers}
        weights = [group[layer]["queries"] for layer in layers]
        kl_delta = math.fsum(deltas[layer]["fixed_support_kl"] * n for layer, n in zip(layers, weights)) / sum(weights)
        coverage_delta = math.fsum(deltas[layer]["coverage"] for layer in layers) / len(layers)
        conditions = {
            "baseline_collapse_reproduced": bool(collapsed),
            "all_collapsed_layers_nonpositive_drop_at_least_10pp": bool(collapsed) and all(deltas[l]["nonpositive_rate"] <= -.10 for l in collapsed),
            "mean_layer_coverage_gain_at_least_001": coverage_delta >= .01,
            "no_layer_coverage_drop_over_001": all(deltas[l]["coverage"] >= -.01 for l in layers),
            "query_weighted_holdout_fixed_kl_delta_at_most_001_nat": kl_delta <= .01,
        }
        candidates.append({"arm": arm["name"], "per_layer_deltas_vs_original": deltas,
                           "mean_layer_coverage_delta": coverage_delta, "query_weighted_holdout_kl_delta": kl_delta,
                           "conditions": conditions, "passes_frozen_engineering_screen": all(conditions.values())})
    return {"reference_arm": reference, "fixed_final_update": 200,
            "reference_collapsed_holdout_layers": collapsed, "candidates": candidates,
            "scope": "Prospective engineering screen only, not statistical significance or LM-quality proof",
            "kl_aggregation": "query-weighted across layers; raw per-layer deltas also retained",
            "if_no_collapse": "Static setting did not reproduce it; does not refute moving-backbone collapse. At most the already allowed single500-update lowLR probe."}


def audit(evidence, protocol_path, artifacts=None):
    evidence, protocol_path = Path(evidence), Path(protocol_path)
    artifacts = Path(artifacts) if artifacts else evidence
    checks = []
    report = {"utc": datetime.now(timezone.utc).isoformat(), "scope": "independent local evidence and tensor-only audit",
              "evidence_directory": str(evidence.resolve()), "artifact_directory": str(artifacts.resolve()),
              "frozen_protocol": str(protocol_path.resolve()), "checks": checks,
              "counts": {"full_lm_loads": 0, "model_forwards": 0, "backwards": 0, "optimizer_updates": 0,
                         "remote_calls": 0, "feature_tensor_loads": 0, "indexer_state_tensor_loads": 0}}

    def check(name, condition, detail=None):
        checks.append({"name": name, "pass": bool(condition), **({"detail": detail} if detail is not None else {})})

    p, summary = read(protocol_path), read(evidence / "summary.json")
    check("executed protocol equals frozen", read(evidence / "protocol.json") == p)
    report["frozen_protocol_sha256"] = sha(protocol_path)
    report["summary_sha256"] = sha(evidence / "summary.json")
    check("complete finite four arm run", summary["status"] == "complete_four_arm_router_diagnostic")
    check("200 update fixed decision", p["decision"]["fixed_final_update"] == 200 and p["decision"]["do_not_select_best_checkpoint"] is True)
    check("wall and estimated cost within declared hard bounds", summary["elapsed_wall_seconds"] <= p["external_hard_timeout_seconds"] and summary["estimated_worker_cost_usd"] <= p["stage_cost_cap_usd"])
    check("checkpoint hash", summary["checkpoint"]["sha256"] == p["checkpoint_sha256"])
    check("checkpoint protocol", summary["checkpoint"]["training_protocol_sha256"] == p["checkpoint_protocol_sha256"])
    check("backbone unchanged receipt", summary["backbone_unchanged"] is True)
    source_checks = {}
    for name, expected in p["expected_source_hashes"].items():
        path = ROOT / name
        source_checks[name] = path.exists() and sha(path) == expected
    check("local frozen source hashes", all(source_checks.values()), source_checks)
    updates, metrics = read_lines(evidence / "updates.jsonl"), read_lines(evidence / "metrics.jsonl")
    report["raw_log_sha256"] = {name: sha(evidence / name) for name in ("updates.jsonl", "metrics.jsonl", "extraction-audit.json")}
    layers = sum(kind == "global" for kind in p["model_config"]["layer_types"])
    samples = {item["window_index"]: item for item in p["sampled_windows"]}
    expected_queries = {split: sum(len(s["query_indices"]) for s in samples.values() if s["split"] == split) for split in ("fit", "holdout")}
    expected_counts = {"backbone_forward_attempts": 24, "backbone_forward_calls": 24,
                       "backbone_backward_calls": 0, "backbone_optimizer_updates": 0,
                       "feature_input_tokens": sum(s["input_tokens"] for s in samples.values()),
                       "feature_word_exposures": sum(s["word_exposures"] for s in samples.values()),
                       "initial_support_indexer_forward_calls": 24 * layers, "teacher_qk_projection_calls": 24 * layers,
                       "teacher_fixed_support_attention_rows": sum(expected_queries.values()) * layers,
                       "teacher_dense_attention_rows": sum(expected_queries.values()) * layers,
                       "indexer_fit_forward_calls": 4 * 200 * layers,
                       "indexer_metric_forward_calls": 4 * 9 * 24 * layers,
                       "indexer_backward_calls": 800, "indexer_optimizer_updates": 800}
    check("all operation counters", summary["counts"] == expected_counts, {"actual": summary["counts"], "expected": expected_counts})
    measurements = {}
    for item in metrics:
        key = item["arm"], item["update"]
        check("no duplicate metric point:" + str(key), key not in measurements)
        measurements[key] = item["metrics"]
        for split, rows in item["metrics"].items():
            check("metric layer keys:" + str(key) + split, sorted(rows) == [str(i) for i in range(layers)])
            for layer, row in rows.items():
                check("metric exact query denominator:" + str(key) + split + layer, row["queries"] == expected_queries[split])
                for name, value in row["metrics"].items():
                    check("finite metric:" + str(key) + split + layer + name,
                          all(math.isfinite(value[k]) for k in ("mean", "min", "max")) and value["min"] <= value["mean"] <= value["max"])
    check("36 evaluation points", len(metrics) == 36)
    check("800 update rows", len(updates) == 800)
    check("sequential arm update order", [row["arm"] for row in updates] == [arm["name"] for arm in p["arms"] for _ in range(200)])
    check("sequential arm evaluation order", [(row["arm"], row["update"]) for row in metrics] == [(arm["name"], update) for arm in p["arms"] for update in range(0, 201, 25)])
    arm_records = {a["name"]: a for a in summary["arms"]}
    check("four unique summary arms", len(summary["arms"]) == len(arm_records) == 4)
    report["final_holdout"] = {}
    for arm in p["arms"]:
        name = arm["name"]
        rows = [row for row in updates if row["arm"] == name]
        check("update1..200:" + name, [r["update"] for r in rows] == list(range(1, 201)))
        check("fixed fit window cycle:" + name, [r["window_index"] for r in rows] == [p["fit_window_order"][i % 12] for i in range(200)])
        for row in rows:
            check("update invariants:" + name + ":" + str(row["update"]),
                  row["lr"] == arm["learning_rate"] and row["score_scale"] == arm["score_scale"]
                  and row["backbone_updates"] == 0 and row["sampled_queries_per_layer"] == len(samples[row["window_index"]]["query_indices"]))
            check("update finite:" + name + ":" + str(row["update"]), all(math.isfinite(row[k]) for k in ("fixed_support_kl", "gradient_norm_before_clip", "parameter_delta_l2", "step_to_parameter_norm_ratio")))
        check("fixed metric cadence:" + name, sorted(u for a, u in measurements if a == name) == list(range(0, 201, 25)))
        recorded = arm_records[name]
        check("shared initialization:" + name, recorded["initial_state_sha256"] == summary["common_initial_indexer_state_sha256"])
        check("arm completion:" + name, recorded["updates_completed"] == 200 and recorded["status"] == "complete_200_router_updates")
        check("initial/final summary equals raw metrics:" + name, recorded["initial_metrics"] == measurements[(name, 0)] and recorded["final_metrics"] == measurements[(name, 200)])
        report["final_holdout"][name] = measurements[(name, 200)]["holdout"]
    # Tensor files contain only cached features or small indexer state dicts.
    import torch
    torch.set_num_threads(1)
    cache_path = artifacts / "fixed-features.pt"
    check("feature cache SHA", sha(cache_path) == summary["feature_cache"]["sha256"])
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    report["counts"]["feature_tensor_loads"] += 1
    check("cache protocol/checkpoint provenance", cached["protocol_sha256"] == canonical(p) and cached["checkpoint_sha256"] == p["checkpoint_sha256"])
    check("cache initial SHA receipt", cached["common_initial_indexer_state_sha256"] == summary["common_initial_indexer_state_sha256"] and cached["initial_parameter_hashes"] == summary["initial_parameter_hashes"])
    check("24 unique cache windows", len(cached["windows"]) == 24 and {w["window_index"] for w in cached["windows"]} == set(samples))
    check("six source fit/holdout balance", Counter((w["source_index"], w["split"]) for w in cached["windows"]) == Counter((s["source_index"], s["split"]) for s in samples.values()))
    supports, targets, tensor_bytes = [], [], 0
    for window in cached["windows"]:
        sample = samples[window["window_index"]]
        check("cache window identity:" + str(window["window_index"]), all(window[k] == sample[k] for k in ("source_index", "split", "query_indices")) and len(window["features"]) == layers)
        for index, feature in enumerate(window["features"]):
            prefix = str(window["window_index"]) + ":layer" + str(index)
            hidden, support, target = feature["hidden"], feature["fixed_support"], feature["target"]
            length, block = sample["input_tokens"], p["model_config"]["block_size"]
            queries = torch.tensor(sample["query_indices"])
            visible = (torch.arange(length // block)[None, :] + 1) * block - 1 <= queries[:, None]
            check("hidden shape/dtype:" + prefix, hidden.shape == (length, p["model_config"]["hidden_size"]) and hidden.dtype == torch.float32)
            check("exact visible/support budget:" + prefix, torch.equal(feature["visible"], visible) and not bool((support & ~visible).any()) and bool((support.sum(-1) == p["model_config"]["selected_complete_blocks"]).all()))
            check("normalized fixed teacher:" + prefix, bool((target >= 0).all()) and bool(((target.sum(-1) - 1).abs() <= 1e-6).all()) and not bool((target.masked_fill(support, 0) != 0).any()))
            probability = feature["dense_token_probability"]
            causal = torch.arange(length)[None, :] <= queries[:, None]
            check("full probability normalization/causality:" + prefix, probability.shape == (len(queries), length) and bool((probability >= 0).all()) and bool(((probability.sum(-1) - 1).abs() <= 1e-6).all()) and not bool((probability.masked_fill(causal, 0) != 0).any()))
            probability_blocks = probability[:, :length // block * block].reshape(len(queries), -1, block)
            mass = probability_blocks.sum(-1) * visible
            tail = (torch.arange(length)[None, :] >= (((queries + 1) // block) * block)[:, None]) & causal
            check("dense mass/tail cache derives from probabilities:" + prefix,
                  torch.allclose(mass, feature["dense_mass"], rtol=1e-6, atol=1e-7) and torch.allclose((probability * tail).sum(-1), feature["tail_mass"], rtol=1e-6, atol=1e-7))
            check("all finite cache tensors:" + prefix, all(bool(torch.isfinite(v).all()) for v in feature.values()))
            tensor_bytes += sum(v.numel() * v.element_size() for v in feature.values())
            supports.append(tensor_sha(support)); targets.append(tensor_sha(target))
    check("cache tensor bytes", tensor_bytes == summary["feature_cache"]["tensor_bytes"] and tensor_bytes <= p["max_feature_cache_bytes"])
    check("cache support/teacher SHA receipts", canonical(supports) == summary["feature_cache"]["fixed_support_sha256"] and canonical(targets) == summary["feature_cache"]["target_sha256"])
    del cached
    report["final_indexer_files"] = []
    for arm in p["arms"]:
        path = artifacts / (arm["name"] + "-indexers.pt")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        report["counts"]["indexer_state_tensor_loads"] += 1
        check("final indexer state/provenance:" + arm["name"], saved["arm"] == arm and saved["updates"] == 200 and saved["protocol_sha256"] == canonical(p) and state_sha(saved["indexers"]) == arm_records[arm["name"]]["final_state_sha256"])
        report["final_indexer_files"].append({"path": str(path), "sha256": sha(path)})
        del saved
    report["frozen_final200_screen"] = fixed_final_screen(p, measurements)
    report["limitations"] = ["Fixed-support teacher equality to frozen main attention is supported by pinned source and prior tiny regression; this audit checks cached probability/support/normalization without rerunning the LM.",
                              "Backbone nonmutation is checked through frozen extraction before/after hash receipts, not a new backbone load.",
                              "Query correlations, repeated windows and shared initialization prohibit treating queries or200updates as independent model repetitions."]
    report["status"] = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    report["checks_passed"] = sum(c["pass"] for c in checks)
    report["checks_total"] = len(checks)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.evidence_dir, args.protocol, args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_passed", "checks_total", "counts")}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
