"""One bounded, fresh sparse indexer-score-scale intervention; frozen training engine unchanged.

The original one-epoch word-based LR horizon remains intact. This launcher
requests its first 500 updates through the existing stop_after_updates API.
Default CLI is validation only. No automatic retry, resume, or follow-up.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_babylm_one_epoch_v0 as original
from scripts.summarize_babylm_one_epoch_v0 import canonical_sha, validate_learning_rate

sha, load, resolve, require, write = original.sha, original.load, original.resolve, original.require, original.write
ADMIN_OVERRIDES = {"max_wall_seconds", "paid_ceiling_usd", "hourly_rate_usd", "stage_spent_usd",
                   "actual_compute_hourly_rate_usd", "hard_timeout_seconds", "execution_hardware"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def derive_child(base, master):
    candidate_scale = 1.0 / math.sqrt(128)
    require(base["model_config"]["index_head_dim"] == 128
            and base["model_config"]["index_score_scale"] == 1.0,
            "This probe requires the original head-dimension 128 and unit score scale")
    require(master["index_score_scale"] == candidate_scale,
            "This probe freezes indexer score scale 1 -> 1/sqrt(128) only")
    require(base["indexer_learning_rate"] == master["indexer_learning_rate"] == 1e-3
            and base["learning_rate"] == 3e-4, "Original indexer and backbone learning rates must remain unchanged")
    require(master["stop_after_updates"] == 500, "This scientific probe is fixed to the first 500 updates")
    require(base["max_updates"] == 1413 and base["max_epochs"] == 1,
            "Base must preserve the original 1413-update one-epoch horizon")
    overrides = master.get("administrative_overrides", {})
    require(set(overrides).issubset(ADMIN_OVERRIDES), "Scientific parameter override is forbidden")
    child = copy.deepcopy(base)
    child["model_config"]["index_score_scale"] = candidate_scale
    child.update(copy.deepcopy(overrides))
    # Retained original notes/metadata are explicitly historical provenance;
    # they do not describe this probe's changed scale, hardware, or stopping point.
    child["probe_execution"] = {"kind": "fresh_indexer_score_scale_prefix", "base_protocol": master["base_protocol"],
        "base_protocol_sha256": master["base_protocol_sha256"], "stop_after_updates": 500,
        "expected_engine_status": "stopped_by_update_limit",
        "original_notes_and_execution_metadata_are_historical": True,
        "scientific_changes": {"model_config.index_score_scale": {"before": 1.0, "after": candidate_scale}},
        "administrative_overrides": copy.deepcopy(overrides),
        "epoch_is_not_complete": True, "resume_weights_used": False}
    require({k for k in set(base) | set(child) if base.get(k) != child.get(k)}
            <= {"model_config", "probe_execution"} | set(overrides),
            "Unexpected derived protocol mutation")
    require({k for k in set(base["model_config"]) | set(child["model_config"])
             if base["model_config"].get(k) != child["model_config"].get(k)} == {"index_score_scale"},
            "Only the nested index_score_scale may change within model_config")
    return child


def validate(master):
    import numpy as np
    require(master.get("schema_version") == 1, "Unsupported probe schema")
    base_path = resolve(master["base_protocol"])
    require(sha(base_path) == master["base_protocol_sha256"], "Base protocol SHA mismatch")
    base = load(base_path)
    original.validate(base)  # Frozen source, epoch ledger, and word horizon, no model calls.
    child = derive_child(base, master)
    require(base["scope"] == "scientific" and base["device"] == "cuda", "Scientific CUDA baseline required")
    require(base["eval_initial"] is True and base["eval_every_updates"] == 250
            and len(base["eval_window_indices"]) == 48, "Original initial/250/500 panel is required")
    pins = master.get("source_sha256", {})
    require(pins.get("scripts/run_babylm_score_scale_probe_v0.py") == sha(__file__), "Probe launcher is not pinned")
    require(all(pins.get(name) == digest for name, digest in base["source_sha256"].items()),
            "Master must retain every original frozen source pin")
    for name, digest in pins.items():
        require(resolve(name).is_relative_to(ROOT) and sha(resolve(name)) == digest, "Source pin mismatch: " + name)
    hard = master["hard_timeout_seconds"]
    require(isinstance(hard, (float, int)) and math.isfinite(hard) and 60 < hard <= 12 * 3600,
            "Finite hard bound <=12h required")
    require(0 < child["max_wall_seconds"] <= hard - 30, "Reserve at least 30 seconds before hard boundary")
    require(math.isfinite(child["hourly_rate_usd"]) and child["hourly_rate_usd"] > 0, "Finite positive hourly bound required")
    require(0 <= child["stage_spent_usd"] < child["paid_ceiling_usd"]
            and child["stage_spent_usd"] + hard / 3600 * child["hourly_rate_usd"] <= child["paid_ceiling_usd"],
            "Hard wall bound does not fit declared cost ceiling")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", master.get("expected_initial_parameter_hashes_sha256", ""))),
            "The frozen original initialization hash-map SHA is required")
    manifest = load(resolve(base["train_manifest"]))
    index = np.load(resolve(base["train_manifest"]).parent / "windows.u64.npy", mmap_mode="r", allow_pickle=False)
    columns = {name: i for i, name in enumerate(manifest["columns"])}
    order = np.random.default_rng(np.random.SeedSequence([base["data_order_seed"], 0])).permutation(len(index))
    chosen = order[:500 * base["windows_per_update"]]
    counts = {"updates": 500, "scientific_updates": 500, "engineering_updates": 0,
              "windows": len(chosen), "forward_calls": len(chosen), "backward_calls": len(chosen)}
    for key, column in (("word_exposures", "word_exposures"), ("input_tokens", "input_tokens"),
                        ("loss_tokens", "next_token_loss_positions")):
        counts[key] = int(index[chosen, columns[column]].sum())
    if "expected_prefix_counts" in master:
        require(master["expected_prefix_counts"] == counts, "Frozen prefix counts differ from actual data order")
    order_sha = hashlib.sha256(chosen.astype("<u8").tobytes()).hexdigest()
    require(master.get("expected_prefix_order_sha256") == order_sha, "Frozen prefix order SHA differs")
    return {"kind": "validation_only_no_model_calls", "child_protocol": child,
            "expected_prefix_counts": counts, "window_ids": [int(i) for i in chosen],
            "prefix_order_sha256": hashlib.sha256(chosen.astype("<u8").tobytes()).hexdigest(),
            "expected_cursor": {"epoch": 0, "position": len(chosen)},
            "expected_initial_parameter_hashes_sha256": master.get("expected_initial_parameter_hashes_sha256"),
            "expected_evaluation_updates": [0, 250, 500], "expected_evaluation_forwards": 144,
            "scientific_changes": {"model_config.index_score_scale": {"before": 1.0, "after": 1.0 / math.sqrt(128)}}}


def migration_gate(master, pod_id):
    path = resolve(master["gate_receipt"])
    require(sha(path) == master["gate_receipt_sha256"], "Migration receipt SHA mismatch")
    gate = load(path)
    require(gate.get("schema_version") == 1 and gate.get("status") == "passed", "Migration gate did not pass")
    require(bool(pod_id) and gate.get("pod_id") == pod_id, "Migration gate belongs to another Pod")
    require(all(gate.get(name) is True for name in ("numerics_passed", "replay_passed", "isolation_passed")),
            "Numerics, original-threshold replay, and isolation must all pass")
    hardware = master["execution_hardware"]
    require(gate.get("execution_hardware") == hardware, "Migration hardware differs")
    require(hardware.get("mig_profile") == "2g.48gb", "This probe execution gate is for the frozen 2g.48gb MIG")
    require(gate.get("source_sha256") == master["source_sha256"], "Migration sources differ from probe sources")
    for name in ("numerics", "replay"):
        receipt = resolve(gate[name + "_receipt"])
        require(sha(receipt) == gate[name + "_receipt_sha256"], name + " receipt changed")
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True, timeout=20).stdout
    require(re.findall(r"\(UUID: (GPU-[0-9a-fA-F-]+)\)", listing) == [hardware["physical_uuid"]],
            "Current physical GPU differs from migration receipt")
    require(re.findall(r"MIG\s+(\S+)\s+Device\s+\d+:\s*\(UUID:\s*(MIG-[0-9a-fA-F-]+)\)", listing)
            == [(hardware["mig_profile"], hardware["mig_uuid"])], "Current MIG differs from migration receipt")
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    require(driver == hardware["driver_version"], "Driver differs from migration receipt")
    return gate


def command(protocol_path, output):
    return [sys.executable, "-u", str(ROOT / "scripts/run_babylm_de_v0.py"),
            "--protocol", str(protocol_path), "--mode", "sparse", "--output-dir", str(output),
            "--stop-after-updates", "500"]


def stop_child(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait(timeout=15)


def supervise(child_command, out, timeout_seconds, state):
    child = None
    try:
        with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr:
            # Keep the child in the controller's process group. An external
            # group timeout therefore covers both; the trainer creates no workers.
            child = subprocess.Popen(child_command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=False)
            state.update(status="running", child_pid=child.pid, child_new_session=False)
            write(out / "stage.json", state)
            code = child.wait(timeout=timeout_seconds)
            require(code == 0, "Training child failed; preserve evidence, no retry")
            return code
    finally:
        stop_child(child)
        if child is not None:
            state["child_returncode"] = child.poll()


def audit_prefix(run_dir, plan):
    """Verify committed prefix accounting and complete optimizer/RNG evidence."""
    from scripts.summarize_babylm_one_epoch_v0 import safe_exp
    run_dir = Path(run_dir)
    summary, protocol = load(run_dir / "summary.json"), load(run_dir / "protocol.json")
    require(summary["status"] == "stopped_by_update_limit", "Probe did not reach its requested prefix boundary")
    require(protocol == plan["child_protocol"], "Executed child protocol differs")
    expected = plan["expected_prefix_counts"]
    for key, value in expected.items():
        require(summary["counts"][key] == value, "Prefix count mismatch: " + key)
    require(summary["cursor"] == plan["expected_cursor"], "Prefix cursor differs")
    updates, evaluations, starts, stops, seen = [], [], [], [], []
    raw_totals = {key: 0 for key in ("word_exposures", "input_tokens", "loss_tokens")}
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        kind = event["type"]
        require(kind not in {"resume", "run_resume", "failure", "run_error", "consumed_without_update"},
                "Unexpected resume, failure, or zero-target event in fresh prefix")
        if kind == "run_start": starts.append(event)
        elif kind == "run_stop": stops.append(event)
        elif kind == "evaluation": evaluations.append(event)
        elif kind == "update":
            require(event["counts"]["updates"] == len(updates) + 1, "Update order differs")
            require(len(event["windows"]) == protocol["windows_per_update"], "Prefix batch is incomplete")
            for row in event["windows"]:
                require(row["epoch"] == 0 and all(row[k] is True for k in
                        ("forward_started", "forward_completed", "backward_started", "backward_completed")), "Incomplete or repeated-epoch training window")
                seen.append(row["window_index"])
                require(row["loss_tokens"] == row["input_tokens"] - 1, "Window target accounting differs")
                for key in raw_totals:
                    raw_totals[key] += row[key]
            for key, value in raw_totals.items():
                require(event["counts"][key] == value, "Cumulative raw-window count differs: " + key)
            require(event["loss_tokens_this_update"] == sum(row["loss_tokens"] for row in event["windows"]),
                    "Update target denominator differs from raw windows")
            for key in ("loss_token_weighted_ce", "loss_token_weighted_aux", "grad_norm", "backbone_grad_norm", "indexer_grad_norm"):
                require(math.isfinite(event[key]), "Nonfinite update statistic: " + key)
            validate_learning_rate(event, protocol, event["counts"]["word_exposures"])
            updates.append(event)
    require(len(starts) == len(stops) == 1 and starts[0]["counts"]["updates"] == 0,
            "Fresh zero-count initialization and one terminal event required")
    require(starts[0]["cursor"] == {"epoch": 0, "position": 0}, "Nonfresh initial cursor")
    if plan.get("expected_initial_parameter_hashes_sha256"):
        require(canonical_sha(starts[0]["initial_parameter_hashes"]) == plan["expected_initial_parameter_hashes_sha256"],
                "Actual fresh initialization differs from frozen baseline")
    require(len(updates) == expected["updates"] and seen == plan["window_ids"] and len(set(seen)) == len(seen),
            "Prefix did not exactly consume frozen window order once")
    require(stops[0]["status"] == "stopped_by_update_limit" and stops[0]["counts"] == summary["counts"],
            "Terminal event/summary mismatch")
    require([e["counts"]["updates"] for e in evaluations] == plan["expected_evaluation_updates"],
            "Initial/interval/endpoint panel missing")
    require(summary["eval_counts"]["forward_calls"] == plan["expected_evaluation_forwards"], "Panel count mismatch")
    for event in evaluations:
        require(event["metrics"]["window_indices"] == protocol["eval_window_indices"], "Panel window order changed")
        require(math.isfinite(event["metrics"]["total"]["nll"]), "Nonfinite panel NLL")
    checkpoint_path = run_dir / "checkpoint.pt"
    require(sha(checkpoint_path) == summary["checkpoint_sha256"], "Final full checkpoint SHA mismatch")
    import torch
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("model_state", "optimizer_state", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"):
        require(key in saved, "Checkpoint missing optimizer/model/RNG component: " + key)
    require(saved["resume_supported"] is True and saved["reason"] == "stopped_by_update_limit"
            and saved["pending_batch"] == [] and saved["active_evaluation"] is None,
            "Final checkpoint is not a committed optimizer/RNG boundary")
    require(saved["counts"] == summary["counts"] and saved["cursor"] == summary["cursor"], "Checkpoint counters differ")
    require(saved["log_sha256"] == sha(run_dir / "events.jsonl"), "Checkpoint event-log SHA differs")
    checkpoint_parts = {key: True for key in ("model_state", "optimizer_state", "python_rng", "numpy_rng", "torch_rng", "cuda_rng")}
    del saved
    tail = updates[-min(100, len(updates)):]
    nt = sum(e["loss_tokens_this_update"] for e in tail)
    ce = math.fsum(e["loss_token_weighted_ce"] * e["loss_tokens_this_update"] for e in tail) / nt
    return {"status": "complete_prefix_audited", "scope": "Fresh sparse 500-update indexer-score-scale intervention; not a complete epoch",
            "counts": summary["counts"], "eval_counts": summary["eval_counts"], "cursor": summary["cursor"],
            "tail": {"first_update": len(updates)-len(tail)+1, "last_update": len(updates), "loss_tokens": nt,
                     "token_weighted_lm_nll": ce, "token_weighted_lm_ppl": safe_exp(ce),
                     "arithmetic_mean_step_lm_ppl": math.fsum(safe_exp(e["loss_token_weighted_ce"]) for e in tail)/len(tail),
                     "scope": "online changing-model training loss, not held-out evaluation"},
            "panel_updates": [e["counts"]["updates"] for e in evaluations],
            "panel_metrics": [e["metrics"] for e in evaluations],
            "checkpoint_sha256": summary["checkpoint_sha256"], "checkpoint_components_present": checkpoint_parts,
            "analysis_forward_calls": 0, "analysis_backward_calls": 0, "analysis_optimizer_updates": 0}


def execute(master_path, digest):
    require(os.name == "posix", "Scientific probe requires Linux")
    master_path = resolve(master_path)
    require(sha(master_path) == digest, "Explicit frozen master SHA required")
    master = load(master_path); plan = validate(master)
    require(master.get("launch_allowed") is True and "expected_prefix_counts" in master, "Unfrozen probe launch/counts")
    out = resolve(master["output_dir"])
    require(out.is_relative_to(ROOT / "results") and not out.exists(), "Fresh results directory required")
    import fcntl
    lock_path = "/tmp/babylm-one-epoch-" + master["execution_hardware"]["mig_uuid"] + ".lock"
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = migration_gate(master, os.environ.get("RUNPOD_POD_ID"))
        out.mkdir(parents=True, exist_ok=False)
        child_path = out / "child-protocol.json"
        write(child_path, plan["child_protocol"])
        state = {"status": "starting", "pid": os.getpid(), "started_utc": utc(), "protocol_sha256": digest,
                 "gate_receipt_sha256": master["gate_receipt_sha256"], "pod_id": gate["pod_id"],
                 "execution_hardware": gate["execution_hardware"], "lock_path": lock_path,
                 "expected_prefix_counts": plan["expected_prefix_counts"], "scientific_changes": plan["scientific_changes"],
                 "no_automatic_resume_retry_or_followup": True, "hard_timeout_seconds": master["hard_timeout_seconds"]}
        start = time.monotonic()
        previous_handler = signal.getsignal(signal.SIGTERM)
        def stopped(signum, frame):
            raise InterruptedError(f"Received signal {signum}; stop prefix supervisor")
        signal.signal(signal.SIGTERM, stopped)
        try:
            supervise(command(child_path, out / "run"), out, master["hard_timeout_seconds"], state)
            audit = audit_prefix(out / "run", plan)
            write(out / "prefix-audit.json", audit)
            state.update(status="complete_prefix_audited", counts=audit["counts"], eval_counts=audit["eval_counts"],
                         checkpoint_sha256=audit["checkpoint_sha256"])
            return 0
        except BaseException as error:
            state.update(status="failed_or_incomplete", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            state.update(finished_utc=utc(), elapsed_wall_seconds=time.monotonic()-start)
            state["estimated_worker_cost_usd"] = state["elapsed_wall_seconds"] / 3600 * plan["child_protocol"]["hourly_rate_usd"]
            state["cost_scope"] = "Worker wall estimate, excludes prior setup/idle and is not invoice"
            write(out / "stage.json", state)
            signal.signal(signal.SIGTERM, previous_handler)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        require(bool(args.protocol_sha256), "--protocol-sha256 required for execution")
        return execute(args.protocol, args.protocol_sha256)
    plan = validate(load(resolve(args.protocol)))
    print(json.dumps({k: v for k, v in plan.items() if k not in {"child_protocol", "window_ids"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
