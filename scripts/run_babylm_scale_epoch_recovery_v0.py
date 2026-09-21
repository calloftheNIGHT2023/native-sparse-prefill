"""Bounded u1000 recovery after an interrupted C1 branch; frozen engine unchanged.

Default CLI only validates. Original evidence is never truncated or unlocked.
The abandoned u1001..1074 work remains a separate physical-compute ledger.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_babylm_scale_epoch_continuation_v0 as c1
from scripts import run_babylm_score_scale_probe_v0 as prefix

sha, load, resolve, require, write, utc = c1.sha, c1.load, c1.resolve, c1.require, c1.write, c1.utc
canonical_sha, tree_sha = c1.canonical_sha, c1.tree_sha
SELF = "scripts/run_babylm_scale_epoch_recovery_v0.py"
ADMIN_FIELDS = c1.ADMIN_FIELDS | {"execution_hardware"}


def derive_child(parent, overrides):
    require(set(overrides) <= ADMIN_FIELDS, "Only explicitly administrative fields may change")
    child = copy.deepcopy(parent)
    child.update(copy.deepcopy(overrides))
    require({k for k in set(parent) | set(child) if parent.get(k) != child.get(k)} <= ADMIN_FIELDS,
            "Scientific recovery protocol changed")
    return child


def inspect_gate(master, old_runtime, panel_indices):
    path = resolve(master["gate_receipt"])
    require(sha(path) == master["gate_receipt_sha256"], "Recovery gate SHA differs")
    gate = load(path)
    require(gate.get("schema_version") == 1 and gate.get("status") == "passed", "Recovery gate has not passed")
    require(gate.get("pod_id") == master["pod_id"] and gate.get("execution_hardware") == master["execution_hardware"],
            "Recovery gate Pod/hardware differs")
    require(gate.get("source_sha256") == master["source_sha256"], "Recovery gate source pins differ")
    require(all(gate.get(k) is True for k in ("numerics_passed", "replay_passed", "isolation_passed")),
            "Original-threshold numerics, checkpoint replay and isolation must pass")
    for kind in ("numerics", "replay"):
        require(sha(resolve(gate[kind + "_receipt"])) == gate[kind + "_receipt_sha256"], "Gate evidence changed: " + kind)
    numerical = load(resolve(gate["numerics_receipt"]))
    from scripts.run_babylm_sparse_parallel_v0 import THRESHOLDS, NUMERICAL_CHECKS
    require(numerical.get("status") == "passed" and numerical.get("passed") is True
            and numerical.get("scope") == "tiny_synthetic_gpu_numerics_engineering_only"
            and numerical.get("thresholds") == THRESHOLDS
            and set(numerical.get("checks", {})) == set(NUMERICAL_CHECKS), "Original seven numerical checks/thresholds did not pass")
    def inspect_nested(value, threshold):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("passed", "finite"): require(item is True, "Nested numerical check failed")
                if key == "threshold": require(item == threshold, "Nested numerical threshold changed")
                if key in ("failed_names", "missing_gradients", "nonfinite_gradients", "zero_gradient_names"):
                    require(item == [], "Numerical failure details are nonempty")
                inspect_nested(item, threshold)
        elif isinstance(value, list):
            for item in value: inspect_nested(item, threshold)
        elif isinstance(value, float): require(math.isfinite(value), "Nonfinite numerical evidence")
    for name, check in numerical["checks"].items():
        require(check.get("passed") is True, "Missing numerical pass: " + name)
        threshold = (THRESHOLDS["cpu_cuda_fp32"] if name.startswith("cpu_cuda_fp32_") else
                     THRESHOLDS["gpu_full_support"] if name.startswith("gpu_full_support_") else
                     THRESHOLDS["checkpoint_replay"] if name.startswith("checkpoint_") else None)
        inspect_nested(check, threshold)
    for k, v in {"model_forward_attempts": 9, "model_forward_calls": 9, "backward_attempts": 9,
                 "backward_calls": 9, "engineering_optimizer_steps": 3, "scientific_optimizer_steps": 0}.items():
        require(numerical["counts"].get(k) == v, "Numerical count differs: " + k)
    nh = numerical["hardware"]
    rows = list(csv.reader(nh["nvidia_smi"]["stdout"].splitlines(), skipinitialspace=True))
    require(nh["nvidia_smi"]["returncode"] == 0 and nh["visible_cuda_device_count"] == 1 and len(rows) == 1
            and rows[0][2] == master["execution_hardware"]["physical_uuid"]
            and rows[0][3] == master["execution_hardware"]["driver_version"], "Numerical receipt belongs to another GPU/driver")
    require(gate.get("recovery_checkpoint_sha256") == master["parent_file_sha256"]["checkpoint.pt"]
            and gate.get("recovery_checkpoint_updates") == gate.get("replayed_panel_update") == 1000,
            "Gate must replay the exact u1000 recovery checkpoint")
    require(gate.get("replayed_window_indices_sha256") == canonical_sha(panel_indices), "Replay panel membership differs")
    runtime = gate["verified_runtime"]
    replay = load(resolve(gate["replay_receipt"]))
    require(replay.get("status") == "passed" and replay.get("replay_passed") is True
            and replay["checkpoint_sha256"] == gate["recovery_checkpoint_sha256"]
            and replay["checkpoint_counts"]["updates"] == 1000
            and replay["checkpoint_last_eval_point"]["updates"] == 1000, "Wrong checkpoint or failed u1000 replay receipt")
    require(replay["verified_runtime"] == runtime and replay["reference_runtime"] == old_runtime
            and replay["execution_hardware"] == master["execution_hardware"], "Replay runtime/hardware provenance differs")
    require(replay["window_indices"] == panel_indices
            and replay["window_indices_sha256"] == canonical_sha(panel_indices), "Actual replay panel membership differs")
    require(replay["reference_events_sha256"] == master["parent_file_sha256"]["events.jsonl"]
            and replay["checkpoint_unchanged"] is True and replay["reference_log_unchanged"] is True,
            "Replay does not bind the unchanged interrupted evidence")
    require(all(master["source_sha256"].get(k) == v for k, v in replay["source_sha256"].items()),
            "Replay source pins differ")
    require(replay["verified_runtime_sha256"] == canonical_sha(runtime)
            and replay["reference_runtime_sha256"] == canonical_sha(old_runtime)
            and replay["runtime_changed_paths"] == ["gpu.uuid"], "Replay runtime hashes or migration scope differ")
    for k, v in {"forward_attempts": 48, "forward_calls": 48, "committed_windows": 48,
                 "backward_calls": 0, "optimizer_updates": 0}.items():
        require(replay["counts"].get(k) == v, "Actual replay count differs: " + k)
    ra = replay["replay_audit"]
    require(ra["tolerance"] == {"aggregate_nll_abs": 1e-6, "window_nll_abs": 1e-5}
            and math.isfinite(ra["aggregate_nll_abs"]) and 0 <= ra["aggregate_nll_abs"] <= 1e-6
            and math.isfinite(ra["max_window_nll_abs"]) and 0 <= ra["max_window_nll_abs"] <= 1e-5
            and ra["attention_counts_exact"] is True and ra["counts_exact"] is True
            and ra["passed"] is True, "Original checkpoint replay thresholds failed or changed")
    nr = numerical["runtime"]
    require(nr["torch"] == runtime["torch"] and nr["numpy"] == runtime["numpy"]
            and nr["cuda_compiled_version"] == runtime["cuda_version"]
            and nr["torch_num_threads"] == runtime["torch_num_threads"]
            and nr["tf32_matmul"] == runtime["cuda_matmul_allow_tf32"]
            and nr["tf32_cudnn"] == runtime["cudnn_allow_tf32"]
            and nr["deterministic_algorithms"] == runtime["deterministic_algorithms"]
            and nr["dtype"] == "float32", "Numerical/replay runtime differs")
    for rel in ("scripts/check_babylm_gpu_numerics_v0.py", "src/babylm_hybrid/config.py",
                "src/babylm_hybrid/model.py", "src/babylm_hybrid/attention.py"):
        matching = [v for k, v in numerical["source_hashes"].items() if k.replace("\\", "/").endswith("/" + rel)]
        require(matching == [master["source_sha256"][rel]], "Numerical source pin differs: " + rel)
    require(set(runtime) == set(old_runtime), "Runtime signature schema changed")
    require({k: v for k, v in runtime.items() if k != "gpu"} == {k: v for k, v in old_runtime.items() if k != "gpu"},
            "Only verified GPU identity may change in runtime")
    hardware = master["execution_hardware"]
    require(hardware["mig_profile"] == "2g.48gb" and runtime["gpu"]["device_index"] == 0
            and runtime["gpu"]["uuid"].removeprefix("GPU-") == hardware["physical_uuid"].removeprefix("GPU-"),
            "Verified runtime does not name the new physical GPU")
    require(runtime["gpu"]["name"] == old_runtime["gpu"]["name"]
            and runtime["gpu"]["capability"] == old_runtime["gpu"]["capability"]
            and runtime["gpu"]["total_memory_bytes"] == old_runtime["gpu"]["total_memory_bytes"],
            "This bounded recovery is restricted to the same GPU type/MIG capacity")
    return gate


def inspect_interruption(source, expected, expected_updates=1000, expected_last_updates=1074):
    source = Path(source).resolve()
    require({"checkpoint.pt", "events.jsonl", "protocol.json"} <= set(expected), "Incomplete parent pins")
    c1.verify_files(source, expected)  # Stale source run.lock is evidence, never removed.
    raw = (source / "events.jsonl").read_bytes()
    require(raw.endswith(b"\n"), "Interrupted log has an incomplete line")
    events = [json.loads(x) for x in raw.splitlines()]
    require([e["event_id"] for e in events] == list(range(1, len(events) + 1)), "Interrupted event sequence differs")
    updates = [e for e in events if e["type"] == "update"]
    require([e["counts"]["updates"] for e in updates] == list(range(1, expected_last_updates + 1)), "Interrupted update sequence differs")
    require(not any(e["type"] in {"failure", "run_error", "consumed_without_update"} for e in events), "Unexpected parent failure event")
    require(not any(e["type"] == "run_stop" and e.get("status") == "epoch_complete" for e in events), "Parent already completed")
    require(events[-1]["type"] == "update", "Expected interruption after the last recorded update")
    return {"source": str(source), "protocol": load(source / "protocol.json"), "raw": raw,
            "events": events, "updates": updates, "expected_updates": expected_updates}


def prepare_recovery(source, target, child_protocol, expected, verified_runtime,
                     expected_updates=1000, expected_last_updates=1074):
    """No model calls. Preserve old history, bind a verified committed prefix.

    Small alternate update limits are only for CPU engineering fixtures; the
    scientific master validator fixes u1000/u1074.
    """
    import torch
    source, target = Path(source).resolve(), Path(target).resolve()
    require(source != target and not source.is_relative_to(target) and not target.is_relative_to(source), "Overlapping source/target")
    require(not target.exists(), "Recovery target must be new")
    parent = inspect_interruption(source, expected, expected_updates, expected_last_updates)
    old_protocol = parent["protocol"]
    require(derive_child(old_protocol, {k: child_protocol[k] for k in ADMIN_FIELDS if k in child_protocol}) == child_protocol,
            "Recovery changed a scientific field")
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=False)
    require(saved["checkpoint_schema_version"] == 1 and saved["resume_supported"] is True
            and saved["pending_batch"] == [] and saved["active_evaluation"] is None, "Checkpoint is not committed/resumable")
    require(saved["counts"] == saved["counters"] and saved["counts"]["updates"] == expected_updates,
            "Wrong committed update")
    require(saved["protocol"] == old_protocol and saved["protocol_sha256"] == canonical_sha(old_protocol), "Parent protocol differs")
    require(saved["source_hashes"] == parent["events"][0]["source_hashes"]
            and saved["data_fingerprint"] == parent["events"][0]["data_fingerprint"]
            and saved["initial_parameter_hashes"] == parent["events"][0]["initial_parameter_hashes"], "Source/data/initialization provenance differs")
    n, raw = saved["log_bytes"], parent["raw"]
    prefix_raw, suffix_raw = raw[:n], raw[n:]
    require(0 < n < len(raw) and prefix_raw.endswith(b"\n") and hashlib.sha256(prefix_raw).hexdigest() == saved["log_sha256"],
            "Checkpoint log prefix is not exact")
    prefix_events = [json.loads(x) for x in prefix_raw.splitlines()]
    suffix_events = [json.loads(x) for x in suffix_raw.splitlines()]
    require(len(prefix_events) == saved["event_id"] and prefix_events[-1]["counts"] == saved["counts"], "Committed event boundary differs")
    require([e["type"] for e in suffix_events] == ["update"] * (expected_last_updates - expected_updates)
            and [e["counts"]["updates"] for e in suffix_events] == list(range(expected_updates + 1, expected_last_updates + 1)),
            "Unexpected post-checkpoint evidence")
    require(set(verified_runtime) == set(saved["runtime"])
            and {k: v for k, v in verified_runtime.items() if k != "gpu"} == {k: v for k, v in saved["runtime"].items() if k != "gpu"},
            "Unverified non-GPU runtime change")
    mutable = {"run_dir", "protocol", "protocol_sha256", "snapshot_state", "runtime"}
    retained = {k: tree_sha(v) for k, v in saved.items() if k not in mutable}
    old_runtime = copy.deepcopy(saved["runtime"])
    target.mkdir(parents=True, exist_ok=False)
    (target / "snapshots").mkdir()
    evidence = target.parent / "interrupted-source-evidence"
    evidence.mkdir(exist_ok=False)
    (evidence / "events.full.jsonl").write_bytes(raw)
    (evidence / "events.after-checkpoint.jsonl").write_bytes(suffix_raw)
    (target / "events.jsonl").write_bytes(prefix_raw)
    require(sha(target / "events.jsonl") == saved["log_sha256"] and sha(evidence / "events.full.jsonl") == expected["events.jsonl"],
            "Recovery log copy differs")
    if "run.lock" in expected:
        shutil.copyfile(source / "run.lock", evidence / "source-run.lock")
        require(sha(evidence / "source-run.lock") == expected["run.lock"], "Stale lock evidence changed")
    copied = {}
    for name, digest in expected.items():
        if name.startswith("snapshots/"):
            dst = target / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, dst)
            require(sha(dst) == digest, "Historical snapshot copy differs")
            copied[name] = digest
    write(evidence / "source-protocol.json", old_protocol)
    write(target / "protocol.json", child_protocol)
    old_state = saved["snapshot_state"]
    new_state = {"models": [], "crossed_milestones": list(old_state["crossed_milestones"]), "final_receipts": []}
    phash = canonical_sha(child_protocol)
    aliases, mapping = [], {}
    for item in old_state["models"]:
        for pk, hk in (("path", "sha256"), ("metadata_path", "metadata_sha256")):
            require(expected.get(item[pk].replace("\\", "/")) == item[hk], "Snapshot missing from parent inventory")
        payload = torch.load(source / item["path"], map_location="cpu", weights_only=False)
        require(payload["protocol"] == old_protocol and payload["protocol_sha256"] == canonical_sha(old_protocol), "Snapshot protocol differs")
        before = {k: tree_sha(v) for k, v in payload.items() if k not in {"protocol", "protocol_sha256"}}
        newpath = Path("snapshots") / (Path(item["path"]).stem + "-r2-administrative.pt")
        payload.update(protocol=copy.deepcopy(child_protocol), protocol_sha256=phash)
        torch.save(payload, target / newpath)
        require({k: tree_sha(v) for k, v in payload.items() if k not in {"protocol", "protocol_sha256"}} == before,
                "Snapshot rebind changed scientific state")
        ni = copy.deepcopy(item)
        ni.update(path=newpath.as_posix(), sha256=sha(target / newpath), protocol_sha256=phash,
                  metadata_path=newpath.with_suffix(".json").as_posix())
        meta = load(source / item["metadata_path"])
        meta.update(path=ni["path"], sha256=ni["sha256"], protocol_sha256=phash)
        write(target / ni["metadata_path"], meta)
        ni["metadata_sha256"] = sha(target / ni["metadata_path"])
        new_state["models"].append(ni)
        mapping[item["path"]] = ni
        aliases.append({"kind": "model_only_administrative_alias", "parent_path": item["path"], "parent_sha256": item["sha256"],
                        "path": ni["path"], "sha256": ni["sha256"], "retained_payload_tree_sha256": before})
        del payload
    for item in old_state["final_receipts"]:
        require(expected.get(item["path"].replace("\\", "/")) == item["sha256"], "Historical final receipt missing")
        meta = load(source / item["path"])
        model = mapping[item["model_path"]]
        newpath = Path("snapshots") / (Path(item["path"]).stem + "-r2-administrative.json")
        meta.update(model_path=model["path"], model_sha256=model["sha256"], protocol_sha256=phash)
        write(target / newpath, meta)
        ni = copy.deepcopy(item)
        ni.update(meta)
        ni.update(path=newpath.as_posix(), sha256=sha(target / newpath))
        new_state["final_receipts"].append(ni)
        aliases.append({"kind": "historical_final_receipt_alias", "parent_path": item["path"], "parent_sha256": item["sha256"],
                        "path": ni["path"], "sha256": ni["sha256"]})
    saved.update(run_dir=str(target), protocol=copy.deepcopy(child_protocol), protocol_sha256=phash,
                 snapshot_state=new_state, runtime=copy.deepcopy(verified_runtime))
    require({k: tree_sha(v) for k, v in saved.items() if k not in mutable} == retained, "Recovery changed retained state")
    torch.save(saved, target / "checkpoint.pt")
    rebound = torch.load(target / "checkpoint.pt", map_location="cpu", weights_only=False)
    require({k: tree_sha(v) for k, v in rebound.items() if k not in mutable} == retained, "Serialized recovery state differs")
    require(all(rebound[k] == saved[k] for k in mutable), "Serialized administrative fields differ")
    c1.verify_files(source, expected)
    abandoned = {k: parent["updates"][-1]["counts"][k] - saved["counts"][k] for k in saved["counts"]}
    receipt = {"schema_version": 1, "status": "verified_checkpoint_prefix_recovery_branch", "utc": utc(),
        "parent_run_dir": str(source), "new_run_dir": str(target), "parent_checkpoint_sha256": expected["checkpoint.pt"],
        "new_checkpoint_sha256": sha(target / "checkpoint.pt"), "parent_protocol_sha256": canonical_sha(old_protocol),
        "new_protocol_sha256": phash, "changed_protocol_fields": {k: {"before": old_protocol.get(k), "after": child_protocol.get(k)}
            for k in set(old_protocol) | set(child_protocol) if old_protocol.get(k) != child_protocol.get(k)},
        "changed_checkpoint_metadata_fields": sorted(mutable), "retained_state_tree_sha256": retained,
        "old_runtime": old_runtime, "verified_new_runtime": verified_runtime,
        "historical_log_prefix_bytes": n, "historical_log_prefix_sha256": saved["log_sha256"],
        "complete_interrupted_log_sha256": expected["events.jsonl"],
        "abandoned_suffix_sha256": hashlib.sha256(suffix_raw).hexdigest(), "abandoned_suffix_bytes": len(suffix_raw),
        "abandoned_update_range": [expected_updates + 1, expected_last_updates], "abandoned_branch_counts": abandoned,
        "unknown_inflight_work_after_last_recorded_update": True, "historical_copied_file_sha256": copied,
        "administrative_snapshot_aliases": aliases, "initial_counts": saved["counts"], "initial_eval_counts": saved["eval_counts"],
        "initial_cursor": saved["cursor"], "initial_elapsed_wall_seconds_retained": saved["elapsed_wall_seconds"],
        "inherited_resume_count": sum(e["type"] == "resume" for e in prefix_events),
        "no_original_lock_or_log_modified": True, "model_forward_calls": 0, "backward_calls": 0, "optimizer_updates": 0}
    write(target.parent / "recovery-branch-receipt.json", receipt)
    return receipt


def validate(master):
    import numpy as np
    require(master.get("schema_version") == 1, "Unsupported recovery schema")
    pm_path = resolve(master["parent_continuation_master"])
    require(sha(pm_path) == master["parent_continuation_master_sha256"], "C1 master SHA differs")
    pm = load(pm_path)
    require(resolve(master["parent_run_dir"]) == resolve(pm["output_dir"]) / "run", "Wrong interrupted parent path")
    parent = inspect_interruption(resolve(master["parent_run_dir"]), master["parent_file_sha256"])
    b2_master_path = resolve(pm["parent_master"])
    require(sha(b2_master_path) == pm["parent_master_sha256"], "B2 master differs")
    b2 = prefix.validate(load(b2_master_path))
    require(parent["protocol"] == c1.derive_child(b2["child_protocol"], pm["administrative_overrides"]), "Interrupted protocol differs from frozen C1")
    audit_path = resolve(master["recovery_audit"])
    require(sha(audit_path) == master["recovery_audit_sha256"], "Recovery evidence audit changed")
    audit = load(audit_path)
    require(audit["checks_passed"] == audit["checks_total"] and audit["checkpoint"]["updates"] == 1000
            and audit["checkpoint"]["sha256"] == master["parent_file_sha256"]["checkpoint.pt"]
            and audit["logs"]["sha256"] == master["parent_file_sha256"]["events.jsonl"]
            and audit["logs"]["last_complete_update"] == 1074, "Recovery audit binds another interruption")
    child = derive_child(parent["protocol"], master["administrative_overrides"])
    epoch = prefix.original.validate(child)
    require(child["model_config"]["index_score_scale"] == 1 / math.sqrt(128), "Scale-only intervention changed")
    require(master["hard_timeout_seconds"] == 5400, "Recovery training has a fixed 5400-second hard ceiling")
    elapsed = audit["checkpoint"]["elapsed_wall_seconds"]
    require(elapsed < child["max_wall_seconds"] <= elapsed + 5340, "Retained cumulative wall horizon is invalid")
    require(child["hourly_rate_usd"] == 1.4 and master["new_training_allocation_usd"] == 2.1
            and master["new_batch_ceiling_usd"] <= 6 and master["new_training_allocation_usd"] <= master["new_batch_ceiling_usd"], "Recovery allocation differs")
    require(0 <= child["stage_spent_usd"] < child["paid_ceiling_usd"] <= 20, "Invalid cumulative cost envelope")
    pins = master["source_sha256"]
    require(all(pins.get(k) == v for k, v in pm["source_sha256"].items()) and pins.get(SELF) == sha(__file__), "Frozen source pins changed")
    for k, v in pins.items(): require(sha(resolve(k)) == v, "Recovery source differs: " + k)
    gate = inspect_gate(master, audit["hardware"]["checkpoint_runtime"], child["eval_window_indices"])
    index = np.load(resolve(child["train_manifest"]).parent / "windows.u64.npy", mmap_mode="r", allow_pickle=False)
    order = np.random.default_rng(np.random.SeedSequence([child["data_order_seed"], 0])).permutation(len(index))
    observed = [w["window_index"] for e in parent["updates"] for w in e["windows"]]
    require(observed == [int(x) for x in order[:17184]], "Interrupted data sequence differs")
    require(canonical_sha(parent["events"][0]["initial_parameter_hashes"]) == b2["expected_initial_parameter_hashes_sha256"], "Original initialization differs")
    return {"kind": "validation_only_no_model_calls", "source": parent["source"], "child_protocol": child,
        "child_protocol_sha256": canonical_sha(child), "verified_runtime": gate["verified_runtime"],
        "expected_epoch_counts": epoch["expected_counts"], "window_ids": [int(x) for x in order],
        "expected_recovery_updates": 413, "expected_recovery_training_forwards": 6598,
        "expected_recovery_training_backwards": 6598, "expected_recovery_evaluation_forwards": 96,
        "expected_panel_updates": [0, 250, 500, 750, 1000, 1250, 1413], "abandoned_branch_counts": audit["logs"]["suffix_counts"]}


def audit_recovery(run_dir, plan, branch, master, digest):
    run_dir = Path(run_dir)
    raw = (run_dir / "events.jsonl").read_bytes()
    n = branch["historical_log_prefix_bytes"]
    require(raw.endswith(b"\n") and hashlib.sha256(raw[:n]).hexdigest() == branch["historical_log_prefix_sha256"], "Recovered historical prefix changed")
    events = [json.loads(x) for x in raw.splitlines()]
    new = [json.loads(x) for x in raw[n:].splitlines()]
    require([e["event_id"] for e in events] == list(range(1, len(events) + 1)), "Recovery event sequence differs")
    require(new and new[0]["type"] == "resume" and new[0]["checkpoint_sha256"] == branch["new_checkpoint_sha256"]
            and new[0]["counts"] == branch["initial_counts"] and new[0]["cursor"] == branch["initial_cursor"], "Unregistered recovery checkpoint")
    require([e["counts"]["updates"] for e in events if e["type"] == "resume"] == [500, 1000], "Expected exactly two transparent resumes")
    require(not any(e["type"] in {"failure", "run_error", "run_resume", "consumed_without_update", "eval_skipped_budget"} for e in events), "Recovery contains incomplete work")
    summary, protocol = load(run_dir / "summary.json"), load(run_dir / "protocol.json")
    require(protocol == plan["child_protocol"] and summary["protocol_sha256"] == canonical_sha(protocol), "Recovery protocol changed")
    require(summary["status"] == "epoch_complete" and summary["cursor"] == {"epoch": 1, "position": 0}, "Epoch not complete")
    updates = [e for e in events if e["type"] == "update"]
    require([e["counts"]["updates"] for e in updates] == list(range(1, 1414))
            and [e["counts"]["updates"] for e in new if e["type"] == "update"] == list(range(1001, 1414)), "Recovery update range differs")
    seen, totals = [], {k: 0 for k in ("input_tokens", "loss_tokens", "word_exposures")}
    for i, e in enumerate(updates, 1):
        require(e["counts"]["scientific_updates"] == i and e["counts"]["engineering_updates"] == 0
                and len(e["windows"]) == (6 if i == 1413 else 16), "Update/accumulation ledger differs")
        for w in e["windows"]:
            require(w["epoch"] == 0 and w["loss_tokens"] == w["input_tokens"] - 1
                    and all(w[k] is True for k in ("forward_started", "forward_completed", "backward_started", "backward_completed")), "Incomplete training window")
            seen.append(w["window_index"])
            for k in totals: totals[k] += w[k]
        require(all(e["counts"][k] == v for k, v in totals.items()) and e["loss_tokens_this_update"] == sum(w["loss_tokens"] for w in e["windows"]), "Window ledger differs")
        require(all(math.isfinite(e[k]) for k in ("loss_token_weighted_ce", "loss_token_weighted_aux", "grad_norm")), "Nonfinite update")
        c1.validate_learning_rate(e, protocol, e["counts"]["word_exposures"])
    require(seen == plan["window_ids"] and len(set(seen)) == 22598, "Final epoch data order differs")
    for k, v in plan["expected_epoch_counts"].items():
        if k != "final_update_windows": require(summary["counts"][k] == v, "Final count differs: " + k)
    require(events[-1]["type"] == "run_stop" and events[-1]["status"] == "epoch_complete"
            and events[-1]["counts"] == summary["counts"] == updates[-1]["counts"], "Terminal counts differ")
    panels = [e for e in events if e["type"] == "evaluation"]
    require([e["counts"]["updates"] for e in panels] == plan["expected_panel_updates"], "Panel cadence differs")
    es = {k: 0 for k in ("forward_calls", "input_tokens", "loss_tokens", "word_exposures")}
    for e in panels:
        mt = e["metrics"]
        require(mt["window_indices"] == protocol["eval_window_indices"] and mt["total"]["forward_calls"] == 48
                and mt["optimizer_updates"] == 0 and mt["grad_enabled_during_model_forward"] is False
                and mt["auxiliary_loss_included_in_nll"] is False and math.isfinite(mt["total"]["nll"]), "Panel scoring differs")
        for k in es: es[k] += mt["total"][k]
    require(summary["eval_counts"] == dict(es, evaluations=7) and es["forward_calls"] == 336, "Evaluation ledger differs")
    finals = [x for x in summary["snapshot_state"]["final_receipts"] if x["stop_reason"] == "epoch_complete" and x["point"]["updates"] == 1413]
    require(len(finals) == 1 and finals[0]["protocol_completion_boundary"] is True, "Unique final1413 receipt missing")
    final = finals[0]
    require(sha(run_dir / "checkpoint.pt") == summary["checkpoint_sha256"] and sha(run_dir / final["path"]) == final["sha256"]
            and sha(run_dir / final["model_path"]) == final["model_sha256"], "Final checkpoint SHA differs")
    recovery = {k: summary["counts"][k] - branch["initial_counts"][k] for k in summary["counts"]}
    recovery_eval = {k: summary["eval_counts"][k] - branch["initial_eval_counts"][k] for k in summary["eval_counts"]}
    require(recovery["updates"] == 413 and recovery["forward_calls"] == recovery["backward_calls"] == 6598 and recovery_eval["forward_calls"] == 96,
            "Actual recovery work differs")
    b2_event = next(e for e in events if e["type"] == "run_stop" and e["counts"]["updates"] == 500)
    retained_c1 = {k: summary["counts"][k] - b2_event["counts"][k] for k in summary["counts"]}
    retained_eval = {k: summary["eval_counts"][k] - b2_event["eval_counts"][k] for k in summary["eval_counts"]}
    physical_lower_bound = {k: summary["counts"][k] + branch["abandoned_branch_counts"][k] for k in summary["counts"]}
    tail = updates[-100:]
    nt = sum(e["loss_tokens_this_update"] for e in tail)
    nll = math.fsum(e["loss_token_weighted_ce"] * e["loss_tokens_this_update"] for e in tail) / nt
    return {"status": "complete_epoch_continuation_audited", "recovery_status": "complete_epoch_after_verified_second_resume",
        "master_protocol_sha256": digest, "protocol_sha256": canonical_sha(protocol), "child_protocol_sha256": canonical_sha(protocol),
        "source_sha256": master["source_sha256"], "counts": summary["counts"], "eval_counts": summary["eval_counts"],
        "incremental_counts": retained_c1, "incremental_eval_counts": retained_eval,
        "incremental_counts_semantics": "Compatibility: retained trajectory above original B2u500, NOT physical work of this recovery or total consumed C1 work",
        "recovery_incremental_counts": recovery, "recovery_incremental_eval_counts": recovery_eval,
        "abandoned_branch_counts": branch["abandoned_branch_counts"], "known_physical_trajectory_counts_lower_bound": physical_lower_bound,
        "unknown_unlogged_inflight_work_excluded_from_lower_bound": True, "authorized_resume_count": 2,
        "historical_prefix_bytes_unchanged": True, "full_interrupted_log_preserved": True,
        "tail": {"first_update": 1314, "last_update": 1413, "loss_tokens": nt, "token_weighted_lm_nll": nll,
                 "token_weighted_lm_ppl": math.exp(nll), "arithmetic_mean_step_lm_ppl": math.fsum(math.exp(e["loss_token_weighted_ce"]) for e in tail) / 100,
                 "scope": "online changing-model training loss, not held-out evaluation"},
        "panel_updates": plan["expected_panel_updates"], "panel_metrics": [e["metrics"] for e in panels],
        "final_model_checkpoint_path": (run_dir / final["model_path"]).relative_to(ROOT).as_posix(),
        "final_model_checkpoint_sha256": final["model_sha256"], "final_model_receipt_path": str(run_dir / final["path"]),
        "final_full_checkpoint_sha256": summary["checkpoint_sha256"], "analysis_model_forward_calls": 0,
        "analysis_backward_calls": 0, "analysis_optimizer_updates": 0,
        "timing_scope": "Different GPU segments and abandoned work remain separate; no sparse-speed conclusion"}


def execute(path, digest):
    started = time.monotonic()
    require(os.name == "posix", "Recovery execution requires validated Linux environment")
    path = resolve(path)
    require(sha(path) == digest, "Explicit recovery master SHA required")
    master = load(path)
    require(master.get("launch_allowed") is True, "Recovery launch not frozen")
    plan = validate(master)
    out = resolve(master["output_dir"])
    require(out.is_relative_to(ROOT / "results") and not out.exists(), "New recovery result directory required")
    import fcntl
    lock_path = "/tmp/babylm-one-epoch-" + master["execution_hardware"]["mig_uuid"] + ".lock"
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        prefix.migration_gate(master, os.environ.get("RUNPOD_POD_ID"))
        out.mkdir(parents=True, exist_ok=False)
        state = {"status": "preparing_recovery_branch", "pid": os.getpid(), "started_utc": utc(),
                 "protocol_sha256": digest, "gate_receipt_sha256": master["gate_receipt_sha256"],
                 "execution_hardware": master["execution_hardware"], "lock_path": lock_path, "no_automatic_retry": True}
        write(out / "stage.json", state)
        previous = signal.getsignal(signal.SIGTERM)
        def stopped(signum, frame): raise InterruptedError("Recovery controller received stop signal")
        signal.signal(signal.SIGTERM, stopped)
        try:
            cp = out / "child-protocol.json"
            write(cp, plan["child_protocol"])
            branch = prepare_recovery(plan["source"], out / "run", plan["child_protocol"], master["parent_file_sha256"], plan["verified_runtime"])
            require(branch["abandoned_branch_counts"] == plan["abandoned_branch_counts"] and branch["inherited_resume_count"] == 1,
                    "Recovery branch lineage differs")
            remaining = 5400 - (time.monotonic() - started)
            require(remaining > 30, "Administrative preparation exhausted recovery deadline")
            prefix.supervise(c1.command(cp, out / "run"), out, remaining, state)
            c1.verify_files(plan["source"], master["parent_file_sha256"])
            audit = audit_recovery(out / "run", plan, branch, master, digest)
            write(out / "continuation-audit.json", audit)
            state.update(status="complete_epoch_continuation_audited", counts=audit["counts"], eval_counts=audit["eval_counts"],
                         recovery_incremental_counts=audit["recovery_incremental_counts"],
                         recovery_incremental_eval_counts=audit["recovery_incremental_eval_counts"],
                         abandoned_branch_counts=audit["abandoned_branch_counts"])
            return 0
        except BaseException as error:
            state.update(status="failed_or_incomplete", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            state.update(finished_utc=utc(), elapsed_wall_seconds=time.monotonic() - started)
            state["estimated_new_worker_cost_usd"] = state["elapsed_wall_seconds"] / 3600 * plan["child_protocol"]["hourly_rate_usd"]
            state["cost_scope"] = "Recovery wrapper wall only; original GPU segment, abandoned work, preflight and idle remain separate"
            write(out / "stage.json", state)
            signal.signal(signal.SIGTERM, previous)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--protocol-sha256")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.execute:
        require(bool(args.protocol_sha256), "Explicit protocol SHA required")
        return execute(args.protocol, args.protocol_sha256)
    plan = validate(load(resolve(args.protocol)))
    print(json.dumps({k: v for k, v in plan.items() if k not in {"child_protocol", "window_ids", "verified_runtime"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
