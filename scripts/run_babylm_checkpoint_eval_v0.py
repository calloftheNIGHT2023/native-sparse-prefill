"""Frozen, append-only checkpoint evaluation; never train or resume a model."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from torch.nn import functional as F
from scripts import prepare_babylm_windows_v0 as window_reader
from src.babylm_hybrid.config import HybridConfig
from src.babylm_hybrid.model import build_model, Qwen3NextGatedDeltaNet
from src.babylm_hybrid.evaluation import POSITION_BINS, _loss_parts, _aggregate_parts, _bucket, _finish

REQUIRED_SOURCES = (
    "scripts/run_babylm_checkpoint_eval_v0.py", "scripts/prepare_babylm_windows_v0.py",
    "src/babylm_hybrid/config.py", "src/babylm_hybrid/model.py",
    "src/babylm_hybrid/attention.py", "src/babylm_hybrid/evaluation.py")
STAT_KEYS = ("valid_tokens", "aux_loss_query_count", "segments", "logical_dense_causal_pairs",
             "logical_kept_pairs", "allocated_main_score_elements", "indexer_score_elements",
             "aux_nonempty_support_query_count", "indexer_zero_score_visible_query_count")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def resolve(value):
    value = Path(value)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def finite(value, name):
    value = float(value)
    require(math.isfinite(value), f"Nonfinite {name}")
    return value


def ppl(nll):
    if nll is None:
        return None
    try:
        return finite(math.exp(nll), "PPL")
    except OverflowError as exc:
        raise ValueError("PPL overflow") from exc


def verify_sources(protocol):
    expected = protocol.get("expected_source_hashes", {})
    require(isinstance(expected, dict) and all(name in expected for name in REQUIRED_SOURCES),
            "Protocol must pin all required evaluation source hashes")
    actual = {}
    for name, digest in expected.items():
        actual[name] = sha(resolve(name))
        require(actual[name] == digest, f"Source SHA mismatch: {name}")
    dependency = Path(inspect.getfile(Qwen3NextGatedDeltaNet))
    dependency_sha = sha(dependency)
    if protocol.get("transformers_gdn_source_sha256") is not None:
        require(dependency_sha == protocol["transformers_gdn_source_sha256"], "Transformers GDN SHA mismatch")
    if protocol.get("enable_routing_diagnostics", False):
        name = "src/babylm_hybrid/routing_diagnostics.py"
        require(name in expected, "Routing diagnostics source must be pinned")
    return actual, {"path": str(dependency), "sha256": dependency_sha}


def verify_manifest(path, digest):
    require(sha(path) == digest, "Development manifest SHA mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    require(manifest.get("status") == "window_index_word_accounting_and_coverage_verified",
            "Unexpected window manifest status")
    require(manifest.get("columns") == window_reader.COLUMNS, "Window columns mismatch")
    artifacts = {}
    for artifact in manifest["artifacts"]:
        target = (path.parent / artifact["path"]).resolve()
        require(target.parent == path.parent, "Window artifact path escapes manifest directory")
        require(sha(target) == artifact["sha256"], f"Window artifact SHA mismatch: {target.name}")
        artifacts[str(target)] = artifact["sha256"]
    require(str((path.parent / "windows.u64.npy").resolve()) in artifacts, "Window index is not hash-pinned")
    for source in manifest["source_summaries"]:
        target = (window_reader.ROOT / source["token_ids_path_relative_to_project"]).resolve()
        require(any(target.is_relative_to(root.resolve()) for root in (window_reader.TOK, window_reader.DEV_TOK))
                and target.name.endswith(".ids.u32"), "Token stream is outside the fixed ledger")
        require(sha(target) == source["token_ids_sha256"], f"Token stream SHA mismatch: {target.name}")
        artifacts[str(target)] = source["token_ids_sha256"]
    return manifest, {"manifest_path": str(path), "manifest_sha256": digest, "artifacts": artifacts,
                      "total_windows": manifest["total_windows"], "tokenizer_sha256": manifest.get("tokenizer_sha256")}


def load_checkpoint_model(protocol, device, dependency_sha):
    path = resolve(protocol["checkpoint_path"])
    require(sha(path) == protocol["checkpoint_sha256"], "Checkpoint SHA mismatch")
    kind = protocol.get("checkpoint_format", "model_only")
    require(kind in {"model_only", "full"}, "checkpoint_format must be model_only or full")
    # Full checkpoints contain pinned, locally produced optimizer/RNG objects.
    # Their optimizer state is NEVER loaded into an optimizer or executed.
    saved = torch.load(path, map_location="cpu", weights_only=(kind == "model_only"))
    require(isinstance(saved, dict) and isinstance(saved.get("model_state"), dict), "Missing model_state")
    require(saved.get("mode") == protocol["mode"], "Checkpoint mode differs from frozen evaluation mode")
    original = saved.get("protocol", {})
    require(canonical_sha(original.get("model_config")) == canonical_sha(protocol["model_config"]),
            "Checkpoint model_config mismatch")
    for key in ("backbone_seed", "indexer_seed"):
        require(original.get(key) == protocol[key], f"Checkpoint {key} mismatch")
    recorded_sources = saved.get("source_hashes", {})
    for name in REQUIRED_SOURCES[1:]:
        matches = [digest for path, digest in recorded_sources.items() if path.replace("\\", "/").endswith(name)]
        require(matches and all(digest == protocol["expected_source_hashes"][name] for digest in matches),
                f"Checkpoint/evaluation source implementation mismatch: {name}")
    dependency_matches = [digest for path, digest in recorded_sources.items()
                          if path.replace("\\", "/").endswith("transformers/models/qwen3_next/modeling_qwen3_next.py")]
    require(dependency_matches and all(digest == dependency_sha for digest in dependency_matches),
            "Checkpoint/evaluation Transformers GDN source mismatch")
    if "checkpoint_protocol_sha256" in protocol:
        require(saved.get("protocol_sha256") == protocol["checkpoint_protocol_sha256"],
                "Checkpoint training protocol SHA mismatch")
    for name, value in saved["model_state"].items():
        require(isinstance(value, torch.Tensor), f"Non-tensor state: {name}")
        if value.is_floating_point():
            require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                    f"Non-FP32 or nonfinite checkpoint state: {name}")
    cfg = HybridConfig(**protocol["model_config"])
    model = build_model(cfg, protocol["mode"], protocol["backbone_seed"], protocol["indexer_seed"])
    model.load_state_dict(saved["model_state"], strict=True)
    provenance = {"path": str(path), "sha256": protocol["checkpoint_sha256"], "format": kind,
                  "training_protocol_sha256": saved.get("protocol_sha256"),
                  "source_hashes_recorded_by_training": saved.get("source_hashes"),
                  "point": saved.get("point"), "counts": saved.get("counts", saved.get("counters")),
                  "optimizer_state_loaded": False}
    del saved
    model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, provenance


def run_evaluation(protocol):
    require(protocol.get("schema_version") == 1, "Unsupported protocol version")
    require(protocol.get("scope") in {"scientific_evaluation", "engineering"}, "Explicit evaluation scope required")
    require(protocol.get("mode") in {"dense", "sparse"}, "Invalid mode")
    require(protocol.get("device") in {"cpu", "cuda"} and protocol.get("dtype") == "float32",
            "Only explicit CPU/CUDA FP32 is supported")
    maximum = finite(protocol["max_wall_seconds"], "max_wall_seconds")
    require(maximum > 0, "Positive wall-time budget required")
    require(type(protocol.get("position_diagnostics", True)) is bool, "Position diagnostics must be bool")
    require(type(protocol.get("enable_routing_diagnostics", False)) is bool, "Routing diagnostics must be bool")
    policy = protocol.get("routing_policy", "learned")
    require(policy in {"learned", "prefix", "local", "random"}, "Invalid routing policy")
    require(policy == "learned" or protocol.get("enable_routing_diagnostics", False),
            "Routing intervention requires explicit diagnostics enable flag")
    output = resolve(protocol["output_dir"])
    output.mkdir(parents=True, exist_ok=False)  # No overwrite and no implicit resume.
    write_json(output / "protocol.json", protocol)
    started, begin = utc(), time.perf_counter()
    counts = {"forward_attempts": 0, "forward_calls": 0, "committed_windows": 0,
              "submitted_input_tokens": 0, "submitted_word_exposures": 0,
              "backward_calls": 0, "optimizer_updates": 0}
    summary = {"schema_version": 1, "status": "running", "scope": protocol["scope"],
               "mode": protocol["mode"], "started_utc": started,
               "protocol_sha256": canonical_sha(protocol), "counts": counts,
               "metric_scope": "heldout_next_token_LM_NLL; auxiliary objective excluded",
               "optimizer_updates": 0, "backward_calls": 0,
               "observed_window_indices": [],
               "timing_scope": "This evaluation runtime is not sparse-training speed evidence.",
               "wall_limit_kind": "cooperative before/after each window and setup; not external hardkill"}
    diagnostic_rows, buckets, total = [], {}, _bucket("all")
    def check_wall():
        if time.perf_counter() - begin >= maximum:
            raise TimeoutError("Frozen evaluation wall-time ceiling reached")
    try:
        actual_sources, dependency = verify_sources(protocol)
        summary.update({"source_hashes": actual_sources, "transformers_gdn_source": dependency})
        manifest_path = resolve(protocol["dev_manifest"])
        manifest, fingerprint = verify_manifest(manifest_path, protocol["dev_manifest_sha256"])
        summary["data_fingerprint"] = fingerprint
        selected = protocol.get("window_indices")
        selected = list(range(manifest["total_windows"])) if selected is None else selected
        require(isinstance(selected, list) and bool(selected) and
                all(type(i) is int and 0 <= i < manifest["total_windows"] for i in selected)
                and len(set(selected)) == len(selected), "Invalid or repeated evaluation window indices")
        summary.update({"window_indices": selected, "requested_windows": len(selected),
                        "full_manifest_requested": protocol.get("window_indices") is None})
        sources = {row["source_index"]: row["source"] for row in manifest["source_summaries"]}
        buckets = {sid: _bucket(name) for sid, name in sources.items()}
        threads = protocol.get("torch_num_threads", 1)
        require(type(threads) is int and threads > 0, "Invalid torch_num_threads")
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        device = torch.device(protocol["device"])
        check_wall()
        model, provenance = load_checkpoint_model(protocol, device, dependency["sha256"])
        summary["checkpoint"] = provenance
        summary["runtime"] = {"torch": torch.__version__, "numpy": np.__version__,
                              "torch_num_threads": torch.get_num_threads(),
                              "device": str(device), "dtype": "float32", "cuda_version": torch.version.cuda,
                              "float32_matmul_precision": torch.get_float32_matmul_precision(),
                              "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                              "cudnn_deterministic": torch.backends.cudnn.deterministic,
                              "cudnn_benchmark": torch.backends.cudnn.benchmark,
                              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                              "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(), "gpu": None}
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            summary["runtime"]["gpu"] = {"device_index": torch.cuda.current_device(), "name": properties.name,
                                          "uuid": str(getattr(properties, "uuid", "unknown")),
                                          "capability": [properties.major, properties.minor],
                                          "total_memory_bytes": properties.total_memory}
        for key, value in protocol.get("expected_runtime", {}).items():
            require(summary["runtime"].get(key) == value, f"Frozen runtime mismatch: {key}")
        context = nullcontext(None)
        if protocol.get("enable_routing_diagnostics", False):
            require(protocol["mode"] == "sparse", "Routing diagnostics require a sparse checkpoint")
            from src.babylm_hybrid.routing_diagnostics import RoutingDiagnostics
            context = RoutingDiagnostics(model, policy=policy, seed=protocol.get("routing_seed", 20260920))
        with (output / "windows.jsonl").open("x", encoding="utf-8") as stream, context as routing, torch.no_grad():
            for item in window_reader.iter_windows(manifest_path, selected, verify_hashes=False):
                check_wall()
                require(item["single_segment"] and item["reset_model_state_before"], "Window is not an independent reset segment")
                inputs = torch.tensor(item["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
                require(inputs.shape[1] == item["input_tokens"] and item["loss_tokens"] == max(0, inputs.shape[1] - 1),
                        "Window next-token accounting mismatch")
                require(not protocol.get("position_diagnostics", True) or inputs.shape[1] <= POSITION_BINS[-1][1],
                        "Frozen position bins support at most 2048 tokens")
                counts["forward_attempts"] += 1
                counts["submitted_input_tokens"] += int(item["input_tokens"])
                counts["submitted_word_exposures"] += int(item["word_exposures"])
                prediction = model(inputs, aux_weight=0.0)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                counts["forward_calls"] += 1
                targets = int(prediction.token_loss_count)
                require(targets == item["loss_tokens"], "Model/window loss-token mismatch")
                loss = finite(prediction.lm_loss.detach().double().item(), "LM loss") if targets else None
                loss_sum = finite(loss * targets, "window NLL sum") if targets else 0.0
                row = {key: int(item[key]) for key in ("window_index", "source_index", "segment_index_in_source",
                       "source_token_start", "source_token_end", "word_exposures", "input_tokens", "loss_tokens")}
                row.update({"source": sources[item["source_index"]], "nll_sum": loss_sum, "nll": loss,
                            "ppl": ppl(loss), "forward_calls": 1, "grad_enabled": torch.is_grad_enabled(),
                            "model_training": model.training, "routing_policy": policy})
                if protocol.get("position_diagnostics", True):
                    logits = prediction.logits
                    require(logits.ndim == 3 and logits.shape[:2] == inputs.shape, "Position diagnostic logits mismatch")
                    per_token = F.cross_entropy(logits[0, :-1], inputs[0, 1:], reduction="none") if targets else logits.new_empty((0,))
                    require(bool(torch.isfinite(per_token).all()), "Nonfinite token loss")
                    exact_sum = float(per_token.detach().double().sum().item())
                    require(math.isclose(exact_sum, loss_sum, rel_tol=1e-5, abs_tol=1e-5), "LM/logit loss disagreement")
                    row["query_history_bins"] = _loss_parts(per_token)
                    row["window_length_bin"] = next(f"{lo}-{hi}" for lo, hi in POSITION_BINS if lo <= inputs.shape[1] <= hi)
                    diagnostic_rows.append(row)
                attention = Counter()
                for stat in prediction.attention_stats:
                    attention.update({key: int(stat[key]) for key in STAT_KEYS if key in stat})
                row["attention_counts"] = dict(attention)
                for bucket in (total, buckets[item["source_index"]]):
                    bucket["windows"] += 1; bucket["forward_calls"] += 1
                    for key in ("word_exposures", "input_tokens", "loss_tokens"):
                        bucket[key] += int(item[key])
                    bucket["zero_target_windows"] += int(targets == 0)
                    bucket["_nll_sums"].append(loss_sum); bucket["_attention_counts"].update(attention)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
                counts["committed_windows"] += 1
                summary["observed_window_indices"].append(int(item["window_index"]))
                check_wall()
            if routing is not None:
                summary["routing_summary"] = routing.summary()
        require(counts["committed_windows"] == len(selected), "Evaluation did not consume all selected windows")
        require(summary["observed_window_indices"] == selected, "Observed evaluation order differs from frozen order")
        require(all(parameter.grad is None for parameter in model.parameters()), "Evaluation created parameter gradients")
        require(verify_sources(protocol)[0] == actual_sources, "Source files changed during evaluation")
        summary["status"] = "evaluation_complete"
    except BaseException as error:
        summary.update({"status": "evaluation_failed", "error_type": type(error).__name__,
                        "error": str(error), "traceback": traceback.format_exc()})
    summary.update({"completed_utc": utc(), "elapsed_wall_seconds": time.perf_counter() - begin,
                    "total": _finish(total), "per_source": {bucket["source"]: _finish(bucket) for bucket in buckets.values()}})
    try:
        summary["total"]["ppl"] = ppl(summary["total"]["nll"])
        for bucket in summary["per_source"].values():
            bucket["ppl"] = ppl(bucket["nll"])
    except ValueError as error:
        summary.update({"status": "evaluation_failed", "error_type": type(error).__name__, "error": str(error)})
    if diagnostic_rows:
        summary["position_diagnostics"] = {"protocol": "babylm-dev-position-v0", "extra_model_forwards": 0,
            "query_history_bins": _aggregate_parts([row["query_history_bins"] for row in diagnostic_rows]),
            "per_source_query_history_bins": {name: _aggregate_parts([row["query_history_bins"] for row in diagnostic_rows if row["source"] == name]) for name in {row["source"] for row in diagnostic_rows}},
            "scope": "Descriptive position bins; not independent samples or proof of long-range dependence"}
    log_path = output / "windows.jsonl"
    summary["windows_jsonl_sha256"] = sha(log_path) if log_path.exists() else None
    summary["observed_window_indices_sha256"] = canonical_sha(summary["observed_window_indices"])
    summary["requested_window_indices_sha256"] = canonical_sha(summary["window_indices"]) if "window_indices" in summary else None
    summary["partial_metrics_only"] = summary["status"] != "evaluation_complete"
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    result = run_evaluation(protocol)
    print(json.dumps({key: result[key] for key in ("status", "counts", "total", "elapsed_wall_seconds")}, allow_nan=False))
    return 0 if result["status"] == "evaluation_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
