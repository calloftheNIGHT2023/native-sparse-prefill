"""Read-only portability replay of the frozen C1 u1000 full checkpoint.

Exactly the original 48 development windows, FP32 and the original absolute
thresholds. No optimizer, backward, training, checkpoint mutation, or automatic
retry. Root must supply a separate 600-second process-group guard and tiny gate.
Default CLI only checks static evidence and source pins; it never accesses CUDA.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_babylm_one_epoch_v0 as util
from scripts.run_babylm_optimization_stage_a_v0 import replay_audit
from scripts.summarize_babylm_one_epoch_v0 import canonical_sha

sha, load, resolve, require, write = util.sha, util.load, util.resolve, util.require, util.write
TOLERANCE = {"aggregate_nll_abs": 1e-6, "window_nll_abs": 1e-5}
REQUIRED_SOURCES = (
    "scripts/replay_babylm_scale_checkpoint_v0.py", "scripts/run_babylm_checkpoint_eval_v0.py",
    "scripts/run_babylm_optimization_stage_a_v0.py", "scripts/run_babylm_one_epoch_v0.py",
    "scripts/summarize_babylm_one_epoch_v0.py", "scripts/prepare_babylm_windows_v0.py",
    "src/babylm_hybrid/config.py", "src/babylm_hybrid/model.py", "src/babylm_hybrid/attention.py",
    "src/babylm_hybrid/evaluation.py", "src/babylm_hybrid/training.py")


def validate(master):
    require(master.get("schema_version") == 1, "Unsupported replay schema")
    require(master.get("reference_update", 1000) == 1000, "Only the frozen u1000 checkpoint may be replayed")
    require(master["hard_timeout_seconds"] == 600 and 0 < master["max_wall_seconds"] <= 540,
            "Replay requires a 600-second outer guard and <=540-second cooperative evaluation")
    require(master.get("replay_tolerance", TOLERANCE) == TOLERANCE, "Original replay thresholds cannot change")
    sources = master["source_sha256"]
    require(all(name in sources for name in REQUIRED_SOURCES), "Required replay/runtime source pins are missing")
    for name, digest in sources.items():
        path = resolve(name)
        require(path.is_relative_to(ROOT) and sha(path) == digest, "Source SHA mismatch: " + name)
    require(sha(resolve(master["checkpoint_path"])) == master["checkpoint_sha256"], "Full checkpoint SHA differs")
    reference_path = resolve(master["reference_events_path"])
    require(sha(reference_path) == master["reference_events_sha256"], "Reference raw log SHA differs")
    raw = reference_path.read_bytes()
    require(raw.endswith(b"\n"), "Reference event log has an incomplete final line")
    events = [json.loads(line) for line in raw.splitlines()]
    require([e["event_id"] for e in events] == list(range(1, len(events) + 1)), "Reference event IDs differ")
    matches = [e for e in events if e["type"] == "evaluation" and e["counts"]["updates"] == 1000]
    require(len(matches) == 1, "Exactly one original u1000 panel is required")
    event = matches[0]
    ppath = resolve(master["training_protocol_path"])
    require(sha(ppath) == master["training_protocol_file_sha256"], "Training protocol file SHA differs")
    training_protocol = load(ppath)
    require(training_protocol["model_config"]["index_head_dim"] == 128
            and training_protocol["model_config"]["index_score_scale"] == 1 / math.sqrt(128)
            and training_protocol["learning_rate"] == 3e-4 and training_protocol["indexer_learning_rate"] == 1e-3,
            "Reference must be the original scale-only scientific protocol")
    metrics = event["metrics"]
    require(metrics["status"] == "nll_evaluation_complete" and metrics["optimizer_updates"] == 0
            and metrics["grad_enabled_during_model_forward"] is False and metrics["auxiliary_loss_included_in_nll"] is False,
            "Original reference is not a completed pure LM panel")
    indices = training_protocol["eval_window_indices"]
    require(metrics["window_indices"] == indices and len(indices) == len(set(indices)) == 48,
            "Original fixed 48-window panel is required")
    old_rows = metrics["position_diagnostics"]["per_window"]
    require([row["window_index"] for row in old_rows] == indices, "Reference per-window labels/order differ")
    require(metrics["total"]["forward_calls"] == metrics["total"]["windows"] == 48
            and metrics["total"]["input_tokens"] == 77755 and metrics["total"]["loss_tokens"] == 77707,
            "Reference panel counts differ")
    require(all(row["loss_tokens"] > 0 and math.isfinite(row["nll_sum"]) for row in old_rows), "Invalid reference row loss")
    require(metrics["manifest_sha256"] == training_protocol["eval_manifest_sha256"], "Reference dev manifest differs")
    if "numerics_receipt" in master:
        require(sha(resolve(master["numerics_receipt"])) == master["numerics_receipt_sha256"], "Separate tiny numerical receipt changed")
    return {"training_protocol": training_protocol, "reference_event": event,
            "reference_event_canonical_sha256": canonical_sha(event), "reference_event_id": event["event_id"],
            "reference_raw_bytes": raw, "reference_events": events, "indices": indices}


def checkpoint_metadata(master, plan):
    """Load trusted local checkpoint objects on CPU; construct no model/optimizer."""
    import torch
    saved = torch.load(resolve(master["checkpoint_path"]), map_location="cpu", weights_only=False)
    require(saved.get("checkpoint_schema_version") == 1 and saved.get("resume_supported") is True,
            "Expected a committed full optimizer/RNG checkpoint")
    require(all(k in saved for k in ("model_state", "optimizer_state", "python_rng", "numpy_rng", "torch_rng", "cuda_rng")),
            "Full checkpoint components are missing")
    require(saved["mode"] == "sparse" and saved["protocol"] == plan["training_protocol"]
            and saved["protocol_sha256"] == canonical_sha(plan["training_protocol"]), "Checkpoint scientific protocol differs")
    event = plan["reference_event"]
    require(saved["counts"] == saved["counters"] == event["counts"] and saved["counts"]["updates"] == 1000,
            "Checkpoint is not exactly the original u1000 model")
    require(saved["cursor"] == event["cursor"] == {"epoch": 0, "position": 16000}
            and saved["eval_counts"] == event["eval_counts"], "Checkpoint/evaluation cursor or counts differ")
    require(saved["pending_batch"] == [] and saved["active_evaluation"] is None
            and saved["last_eval_point"] == {"updates": 1000, "word_exposures": saved["counts"]["word_exposures"]},
            "Checkpoint is not the committed post-evaluation u1000 state")
    log_prefix = plan["reference_raw_bytes"][:saved["log_bytes"]]
    require(hashlib.sha256(log_prefix).hexdigest() == saved["log_sha256"] and log_prefix.endswith(b"\n"),
            "Checkpoint log-prefix bytes/SHA mismatch")
    prefix_events = [json.loads(line) for line in log_prefix.splitlines()]
    require(len(prefix_events) == saved["event_id"] and event["event_id"] <= saved["event_id"]
            and prefix_events[-1]["counts"] == saved["counts"], "Checkpoint committed event boundary differs")
    metadata = {"checkpoint_counts": saved["counts"], "checkpoint_cursor": saved["cursor"],
                "checkpoint_protocol_sha256": saved["protocol_sha256"], "checkpoint_log_bytes": saved["log_bytes"],
                "checkpoint_log_sha256": saved["log_sha256"], "checkpoint_event_id": saved["event_id"],
                "checkpoint_eval_counts": saved["eval_counts"], "checkpoint_last_eval_point": saved["last_eval_point"],
                "reference_runtime": saved["runtime"], "reference_runtime_sha256": canonical_sha(saved["runtime"]),
                "initial_parameter_hashes_sha256": canonical_sha(saved["initial_parameter_hashes"])}
    del saved
    return metadata


def validate_runtime_change(old, new):
    require(set(old) == set(new), "Runtime signature schema changed")
    require({k: v for k, v in old.items() if k != "gpu"} == {k: v for k, v in new.items() if k != "gpu"},
            "Runtime software or determinism changed")
    require(isinstance(old["gpu"], dict) and isinstance(new["gpu"], dict) and set(old["gpu"]) == set(new["gpu"]),
            "GPU runtime signature schema changed")
    require({k: v for k, v in old["gpu"].items() if k != "uuid"} == {k: v for k, v in new["gpu"].items() if k != "uuid"},
            "GPU name, capability, memory, or device index changed")
    return ["gpu.uuid"] if old["gpu"]["uuid"] != new["gpu"]["uuid"] else []


def configure_and_identify(hardware, reference_runtime):
    import torch
    from src.babylm_hybrid.training import _runtime_signature
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one visible CUDA device required")
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True, timeout=20).stdout
    require(re.findall(r"\(UUID: (GPU-[0-9a-fA-F-]+)\)", listing) == [hardware["physical_uuid"]], "Physical GPU differs")
    require(re.findall(r"MIG\s+(\S+)\s+Device\s+\d+:\s*\(UUID:\s*(MIG-[0-9a-fA-F-]+)\)", listing)
            == [(hardware["mig_profile"], hardware["mig_uuid"])], "MIG identity differs")
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    require(driver == hardware["driver_version"] and hardware["mig_profile"] == "2g.48gb", "MIG profile or driver differs")
    runtime = _runtime_signature("cuda")
    require(uuid.UUID(runtime["gpu"]["uuid"].removeprefix("GPU-").removeprefix("MIG-"))
            == uuid.UUID(hardware["physical_uuid"].removeprefix("GPU-")), "CUDA runtime physical UUID differs")
    changes = validate_runtime_change(reference_runtime, runtime)
    return runtime, changes


def evaluation_protocol(master, plan, runtime):
    p = plan["training_protocol"]
    return {"schema_version": 1, "scope": "scientific_evaluation", "mode": "sparse", "device": "cuda", "dtype": "float32",
            "checkpoint_path": master["checkpoint_path"], "checkpoint_sha256": master["checkpoint_sha256"],
            "checkpoint_format": "full", "checkpoint_protocol_sha256": canonical_sha(p),
            "model_config": copy.deepcopy(p["model_config"]), "backbone_seed": p["backbone_seed"], "indexer_seed": p["indexer_seed"],
            "dev_manifest": p["eval_manifest"], "dev_manifest_sha256": p["eval_manifest_sha256"],
            "window_indices": plan["indices"], "position_diagnostics": True, "enable_routing_diagnostics": False,
            "routing_policy": "learned", "torch_num_threads": 1, "expected_runtime": runtime,
            "expected_source_hashes": master["source_sha256"], "transformers_gdn_source_sha256": master["transformers_gdn_source_sha256"],
            "max_wall_seconds": master["max_wall_seconds"], "output_dir": str(resolve(master["output_dir"]) / "evaluation")}


def execute(master_path, expected_sha):
    require(os.name == "posix", "CUDA portability replay execution requires Linux")
    path = resolve(master_path)
    require(sha(path) == expected_sha, "Explicit frozen master SHA required")
    master = load(path)
    require(master.get("launch_allowed") is True, "Replay launch is not frozen")
    output = resolve(master["output_dir"])
    require(output.is_relative_to(ROOT / "results") and not output.exists(), "Fresh replay results directory required")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "status": "running", "replay_passed": False, "master_protocol_sha256": expected_sha,
               "started_utc": util.datetime.now(util.timezone.utc).isoformat(), "reference_update": 1000,
               "source_sha256": master["source_sha256"], "checkpoint_sha256": master["checkpoint_sha256"],
               "reference_events_sha256": master["reference_events_sha256"], "execution_hardware": master["execution_hardware"],
               "tolerance": TOLERANCE, "outer_hard_limit_seconds_required": 600,
               "counts": {"forward_attempts": 0, "forward_calls": 0, "committed_windows": 0, "backward_calls": 0, "optimizer_updates": 0},
               "not_a_training_or_full_dev_result": True, "automatic_retry_or_training": False}
    write(output / "replay-receipt.json", receipt)
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGTERM)
    def stopped(signum, frame): raise InterruptedError("Portability replay interrupted")
    signal.signal(signal.SIGTERM, stopped)
    try:
        plan = validate(master)
        receipt["reference_event_canonical_sha256"] = plan["reference_event_canonical_sha256"]
        receipt["window_indices"] = plan["indices"]
        receipt["window_indices_sha256"] = canonical_sha(plan["indices"])
        receipt.update(checkpoint_metadata(master, plan))
        import fcntl
        lock_path = "/tmp/babylm-one-epoch-" + master["execution_hardware"]["mig_uuid"] + ".lock"
        with open(lock_path, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt["lock_path"] = lock_path
            runtime, changed = configure_and_identify(master["execution_hardware"], receipt["reference_runtime"])
            receipt.update(verified_runtime=runtime, verified_runtime_sha256=canonical_sha(runtime), runtime_changed_paths=changed)
            from scripts import run_babylm_checkpoint_eval_v0 as evaluator
            child = evaluation_protocol(master, plan, runtime)
            require(time.monotonic() - started < 570, "Setup exhausted the outer replay deadline")
            child["max_wall_seconds"] = min(child["max_wall_seconds"], 570 - (time.monotonic() - started))
            write(output / "evaluation-protocol.json", child)
            summary = evaluator.run_evaluation(child)
            receipt["counts"] = summary["counts"]
            require(summary["status"] == "evaluation_complete", "Evaluator failed: " + str(summary.get("error")))
            require(summary["counts"]["forward_attempts"] == summary["counts"]["forward_calls"] == summary["counts"]["committed_windows"] == 48
                    and summary["backward_calls"] == summary["optimizer_updates"] == 0, "Replay model-call budget differs")
            rows = [json.loads(line) for line in (output / "evaluation/windows.jsonl").read_text(encoding="utf-8").splitlines()]
            require(all(row["grad_enabled"] is False and row["model_training"] is False for row in rows), "Replay enabled training or gradients")
            receipt["replay_audit"] = replay_audit(plan["reference_event"]["metrics"], summary, rows, TOLERANCE)
            from src.babylm_hybrid.training import _runtime_signature
            require(_runtime_signature("cuda") == runtime, "Runtime changed during the replay")
            require(sha(resolve(master["checkpoint_path"])) == master["checkpoint_sha256"], "Original checkpoint changed during replay")
            require(sha(resolve(master["reference_events_path"])) == master["reference_events_sha256"], "Original raw log changed during replay")
            receipt.update(status="passed", replay_passed=True, code_sha256=canonical_sha(summary["source_hashes"]),
                           data_fingerprint_sha256=canonical_sha(summary["data_fingerprint"]), data_fingerprint=summary["data_fingerprint"],
                           checkpoint_unchanged=True, reference_log_unchanged=True,
                           evaluator_protocol_sha256=canonical_sha(child), evaluator_summary_sha256=sha(output / "evaluation/summary.json"),
                           per_window_jsonl_sha256=sha(output / "evaluation/windows.jsonl"), actual_total=summary["total"],
                           checkpoint_optimizer_loaded=False, checkpoint_rng_restored=False)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    except BaseException as error:
        receipt.update(status="failed", replay_passed=False, error_type=type(error).__name__, error=str(error))
        # Preserve the evaluator's physical call counters even if a later check failed.
        summary_path = output / "evaluation/summary.json"
        if summary_path.exists(): receipt["counts"] = load(summary_path)["counts"]
    finally:
        receipt.update(completed_utc=util.datetime.now(util.timezone.utc).isoformat(), elapsed_wall_seconds=time.monotonic() - started)
        write(output / "replay-receipt.json", receipt)
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps({k: receipt[k] for k in ("status", "replay_passed", "counts", "elapsed_wall_seconds")}, allow_nan=False))
    return 0 if receipt["replay_passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        require(bool(args.protocol_sha256), "Explicit --protocol-sha256 required")
        return execute(args.protocol, args.protocol_sha256)
    plan = validate(load(resolve(args.protocol)))
    print(json.dumps({"status": "static_validation_passed", "checkpoint_metadata_loaded": False,
                      "reference_update": 1000, "reference_event_id": plan["reference_event_id"],
                      "reference_event_canonical_sha256": plan["reference_event_canonical_sha256"],
                      "required_forward_calls_if_executed": 48, "model_calls": 0, "cuda_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
