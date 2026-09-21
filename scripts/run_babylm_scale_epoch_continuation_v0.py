"""Administrative branch of the committed scale-only u500 prefix; frozen engine.

Default CLI validates only. Execution creates a new directory, preserves the
entire original event prefix and historical snapshots, explicitly rebinds only
administrative checkpoint provenance, then uses the engine's strict resume API.
No model forward is performed by this controller. No retry or next experiment.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_babylm_score_scale_probe_v0 as prefix
from scripts.summarize_babylm_one_epoch_v0 import canonical_sha, validate_learning_rate

sha, load, resolve, require, write, utc = prefix.sha, prefix.load, prefix.resolve, prefix.require, prefix.write, prefix.utc
ADMIN_FIELDS = {"max_wall_seconds", "paid_ceiling_usd", "hourly_rate_usd", "stage_spent_usd", "hard_timeout_seconds"}


def tree_sha(value):
    """Type-sensitive state digest; CPU tensor bytes, no model construction."""
    import numpy as np
    import torch
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(b"tensor" + str(x.dtype).encode() + repr(tuple(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(b"ndarray" + str(x.dtype).encode() + repr(x.shape).encode() + x.tobytes())
        elif isinstance(x, dict):
            h.update(b"dict{")
            for key in sorted(x, key=lambda k: (type(k).__name__, repr(k))):
                visit(key); visit(x[key])
            h.update(b"}")
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode() + b"[")
            for item in x: visit(item)
            h.update(b"]")
        else:
            h.update(type(x).__name__.encode() + b":" + repr(x).encode("utf-8") + b";")
    visit(value)
    return h.hexdigest()


def derive_child(parent_protocol, administrative_overrides):
    require(set(administrative_overrides) <= ADMIN_FIELDS, "Only administrative wall/cost fields may change")
    child = copy.deepcopy(parent_protocol)
    child.update(copy.deepcopy(administrative_overrides))
    require({k for k in set(child) | set(parent_protocol) if child.get(k) != parent_protocol.get(k)} <= ADMIN_FIELDS,
            "Scientific protocol changed")
    return child


def verify_files(directory, expected):
    directory = Path(directory).resolve()
    require(bool(expected), "Explicit parent file pins required")
    for name, digest in expected.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory) and path.is_file() and not path.is_symlink(), "Unsafe parent evidence path: " + name)
        require(sha(path) == digest, "Parent evidence SHA mismatch: " + name)


def inspect_parent(source, expected, expected_updates=500):
    """Read only JSON and raw logs, without loading the model checkpoint."""
    source = Path(source).resolve()
    require(not (source / "run.lock").exists(), "Parent run lock remains present")
    require({"events.jsonl", "protocol.json", "summary.json", "checkpoint.pt"} <= set(expected), "Incomplete parent evidence pins")
    verify_files(source, expected)
    protocol, summary = load(source / "protocol.json"), load(source / "summary.json")
    raw = (source / "events.jsonl").read_bytes()
    require(raw.endswith(b"\n"), "Incomplete parent terminal event")
    events = [json.loads(line) for line in raw.splitlines()]
    require([e["event_id"] for e in events] == list(range(1, len(events) + 1)), "Parent event sequence differs")
    require(not any(e["type"] in {"resume", "run_resume", "failure", "run_error", "consumed_without_update"} for e in events),
            "Parent must be a fresh, successful, unreplayed prefix")
    updates = [e for e in events if e["type"] == "update"]
    stops = [e for e in events if e["type"] == "run_stop"]
    starts = [e for e in events if e["type"] == "run_start"]
    require(len(updates) == expected_updates and len(stops) == len(starts) == 1, "Parent prefix length differs")
    require(summary["status"] == stops[0]["status"] == "stopped_by_update_limit", "Parent prefix did not finish cleanly")
    require(summary["counts"] == updates[-1]["counts"] == stops[0]["counts"], "Parent terminal counters differ")
    require(summary["checkpoint_sha256"] == expected["checkpoint.pt"], "Parent checkpoint receipt differs")
    require(summary["protocol_sha256"] == canonical_sha(protocol) == starts[0]["protocol_sha256"], "Parent protocol differs")
    require(starts[0]["counts"]["updates"] == 0 and starts[0]["cursor"] == {"epoch": 0, "position": 0}, "Parent was not fresh")
    for number, event in enumerate(updates, 1):
        require(event["counts"]["updates"] == number, "Parent update sequence differs")
        validate_learning_rate(event, protocol, event["counts"]["word_exposures"])
        require(math.isfinite(event["loss_token_weighted_ce"]), "Nonfinite parent LM NLL")
    for item in summary["snapshot_state"]["models"]:
        for key, hash_key in (("path", "sha256"), ("metadata_path", "metadata_sha256")):
            require(expected.get(item[key].replace("\\", "/")) == item[hash_key], "Parent snapshot omitted from frozen inventory")
    for item in summary["snapshot_state"]["final_receipts"]:
        require(expected.get(item["path"].replace("\\", "/")) == item["sha256"], "Parent final receipt omitted from inventory")
    return {"protocol": protocol, "summary": summary, "events": events, "updates": updates,
            "log_bytes": len(raw), "log_sha256": sha(source / "events.jsonl"), "source": str(source)}


def prepare_branch(source, target, child_protocol, expected, expected_updates=500):
    """Create an explicit administrative resume copy; never mutate source files.

    Historical event/snapshot bytes remain unchanged under their original paths.
    New administrative snapshot aliases have separate names and recorded lineage.
    Only the aliases are placed in the engine's new-protocol snapshot_state.
    """
    import torch
    source, target = Path(source).resolve(), Path(target).resolve()
    require(source != target and not source.is_relative_to(target) and not target.is_relative_to(source), "Source/target evidence directories overlap")
    require(not target.exists(), "Branch target must be new")
    parent = inspect_parent(source, expected, expected_updates)
    old_protocol, summary = parent["protocol"], parent["summary"]
    require(derive_child(old_protocol, {k: child_protocol[k] for k in ADMIN_FIELDS if k in child_protocol}) == child_protocol,
            "Branch changed a scientific protocol field")
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=False)
    require(saved.get("resume_supported") is True and saved["checkpoint_schema_version"] == 1,
            "Parent checkpoint is not resumable")
    require(saved["reason"] == "stopped_by_update_limit" and saved["pending_batch"] == [] and saved["active_evaluation"] is None,
            "Parent is not a committed optimizer/RNG boundary")
    require(saved["protocol"] == old_protocol and saved["protocol_sha256"] == canonical_sha(old_protocol), "Checkpoint parent protocol differs")
    require(saved["counts"] == saved["counters"] == summary["counts"] and saved["cursor"] == summary["cursor"], "Checkpoint parent counters differ")
    require(saved["log_bytes"] == parent["log_bytes"] and saved["log_sha256"] == parent["log_sha256"], "Checkpoint log commitment differs")
    require(saved["event_id"] == len(parent["events"]) and saved["snapshot_state"] == summary["snapshot_state"], "Checkpoint event/snapshot state differs")
    mutable = {"run_dir", "protocol", "protocol_sha256", "snapshot_state"}
    retained_before = {key: tree_sha(value) for key, value in saved.items() if key not in mutable}
    target.mkdir(parents=True, exist_ok=False)
    (target / "snapshots").mkdir(exist_ok=False)
    copied = {}
    for name in expected:
        if name == "events.jsonl" or name.startswith("snapshots/"):
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, destination)
            require(sha(destination) == expected[name], "Historical copied evidence SHA differs")
            copied[name] = expected[name]
    write(target / "parent-protocol.json", old_protocol)
    write(target / "parent-summary.json", summary)
    write(target / "protocol.json", child_protocol)
    old_state = saved["snapshot_state"]
    new_state = {"models": [], "crossed_milestones": list(old_state["crossed_milestones"]), "final_receipts": []}
    new_phash = canonical_sha(child_protocol)
    aliases, mapping = [], {}
    for item in old_state["models"]:
        original_path = source / item["path"]
        original_meta = load(source / item["metadata_path"])
        payload = torch.load(original_path, map_location="cpu", weights_only=False)
        require(payload["protocol_sha256"] == canonical_sha(old_protocol) and payload["protocol"] == old_protocol,
                "Historical snapshot protocol mismatch")
        tensor_sha = tree_sha(payload["model_state"])
        new_path = Path("snapshots") / (original_path.stem + "-c1-administrative.pt")
        new_meta_path = new_path.with_suffix(".json")
        payload["protocol"], payload["protocol_sha256"] = copy.deepcopy(child_protocol), new_phash
        torch.save(payload, target / new_path)
        require(tree_sha(payload["model_state"]) == tensor_sha, "Rebinding changed snapshot model tensors")
        new_item = copy.deepcopy(item)
        new_item.update(path=new_path.as_posix(), sha256=sha(target / new_path), protocol_sha256=new_phash,
                        metadata_path=new_meta_path.as_posix())
        new_meta = copy.deepcopy(original_meta)
        new_meta.update(path=new_item["path"], sha256=new_item["sha256"], protocol_sha256=new_phash)
        write(target / new_meta_path, new_meta)
        new_item["metadata_sha256"] = sha(target / new_meta_path)
        new_state["models"].append(new_item)
        mapping[item["path"]] = new_item
        aliases.append({"kind": "model_only_administrative_alias", "parent_path": item["path"], "parent_sha256": item["sha256"],
                        "path": new_item["path"], "sha256": new_item["sha256"], "model_state_tree_sha256_unchanged": tensor_sha,
                        "parent_metadata_sha256": item["metadata_sha256"], "metadata_sha256": new_item["metadata_sha256"]})
        del payload
    for item in old_state["final_receipts"]:
        original_receipt = load(source / item["path"])
        model = mapping[item["model_path"]]
        new_path = Path("snapshots") / (Path(item["path"]).stem + "-c1-administrative.json")
        new_receipt = copy.deepcopy(original_receipt)
        new_receipt.update(model_path=model["path"], model_sha256=model["sha256"], protocol_sha256=new_phash)
        write(target / new_path, new_receipt)
        new_item = copy.deepcopy(item)
        new_item.update(new_receipt)
        new_item.update(path=new_path.as_posix(), sha256=sha(target / new_path))
        new_state["final_receipts"].append(new_item)
        aliases.append({"kind": "historical_final_receipt_administrative_alias", "parent_path": item["path"],
                        "parent_sha256": item["sha256"], "path": new_item["path"], "sha256": new_item["sha256"]})
    saved.update(run_dir=str(target), protocol=copy.deepcopy(child_protocol), protocol_sha256=new_phash, snapshot_state=new_state)
    retained_after = {key: tree_sha(value) for key, value in saved.items() if key not in mutable}
    require(retained_before == retained_after, "Administrative branch changed retained training state")
    torch.save(saved, target / "checkpoint.pt")
    # Read back the actual newly serialized artifact rather than trusting memory.
    rebound = torch.load(target / "checkpoint.pt", map_location="cpu", weights_only=False)
    require({key: tree_sha(value) for key, value in rebound.items() if key not in mutable} == retained_before,
            "Serialized branch changed retained training state")
    require(rebound["run_dir"] == str(target) and rebound["protocol"] == child_protocol and rebound["protocol_sha256"] == new_phash,
            "Serialized administrative binding mismatch")
    require(rebound["snapshot_state"] == new_state, "Serialized snapshot binding mismatch")
    verify_files(source, expected)
    receipt = {"schema_version": 1, "status": "administrative_branch_verified", "utc": utc(),
               "parent_run_dir": str(source), "new_run_dir": str(target), "parent_checkpoint_sha256": expected["checkpoint.pt"],
               "new_checkpoint_sha256": sha(target / "checkpoint.pt"), "parent_protocol_sha256": canonical_sha(old_protocol),
               "new_protocol_sha256": new_phash, "changed_protocol_fields": {key: {"before": old_protocol.get(key), "after": child_protocol.get(key)}
                    for key in set(old_protocol) | set(child_protocol) if old_protocol.get(key) != child_protocol.get(key)},
               "changed_checkpoint_metadata_fields": sorted(mutable), "retained_state_tree_sha256": retained_before,
               "historical_log_prefix_bytes": parent["log_bytes"], "historical_log_prefix_sha256": parent["log_sha256"],
               "historical_copied_file_sha256": copied, "administrative_snapshot_aliases": aliases,
               "initial_committed_updates": summary["counts"]["updates"], "initial_counts": summary["counts"],
               "initial_eval_counts": summary["eval_counts"], "initial_cursor": summary["cursor"],
               "initial_elapsed_wall_seconds_retained": saved["elapsed_wall_seconds"],
               "administrative_rebinding_is_not_new_scientific_initialization": True,
               "inherited_prefix_metadata_is_historical_not_a_new_500_step_limit": True,
               "model_forward_calls": 0, "backward_calls": 0, "optimizer_updates": 0}
    write(target.parent / "administrative-branch-receipt.json", receipt)
    return receipt


def validate(master):
    """Validate the frozen control and parent ledger; no checkpoint/model load."""
    import numpy as np
    require(master.get("schema_version") == 1, "Unsupported continuation schema")
    parent_master_path = resolve(master["parent_master"])
    require(sha(parent_master_path) == master["parent_master_sha256"], "Parent master SHA differs")
    pm = load(parent_master_path)
    prefix_plan = prefix.validate(pm)
    source = resolve(master["parent_run_dir"])
    require(source == resolve(pm["output_dir"]) / "run", "Parent evidence path differs from frozen B2 output")
    parent = inspect_parent(source, master["parent_file_sha256"])
    require(parent["protocol"] == prefix_plan["child_protocol"], "Executed B2 protocol differs from frozen parent")
    require(canonical_sha(parent["events"][0]["initial_parameter_hashes"]) == pm["expected_initial_parameter_hashes_sha256"],
            "B2 initialization hash differs")
    seen = [w["window_index"] for e in parent["updates"] for w in e["windows"]]
    require(seen == prefix_plan["window_ids"], "B2 scientific prefix order differs")
    for key, value in prefix_plan["expected_prefix_counts"].items():
        require(parent["summary"]["counts"][key] == value, "B2 expected prefix count differs: " + key)
    require([e["counts"]["updates"] for e in parent["events"] if e["type"] == "evaluation"] == [0, 250, 500],
            "B2 historical panels differ")
    child = derive_child(parent["protocol"], master["administrative_overrides"])
    epoch_plan = prefix.original.validate(child)
    require(child["max_updates"] == 1413 and child["max_epochs"] == 1 and child["model_config"]["index_score_scale"] == 1 / math.sqrt(128),
            "Continuation must finish only the original scale-only epoch")
    require(master["hard_timeout_seconds"] == 10800, "Continuation training hard bound must remain three hours")
    elapsed = float(parent["summary"]["elapsed_wall_seconds"])
    require(elapsed < child["max_wall_seconds"] <= elapsed + master["hard_timeout_seconds"] - 60,
            "Cooperative wall bound must use retained cumulative elapsed with a hard-limit reserve")
    rate, allocation = child["hourly_rate_usd"], master["new_training_allocation_usd"]
    require(math.isfinite(rate) and rate > 0 and master["hard_timeout_seconds"] / 3600 * rate <= allocation,
            "New training hard bound exceeds its allocation")
    require(0 < allocation <= master["new_batch_ceiling_usd"] <= 6,
            "New training allocation must fit the frozen six-dollar train/eval batch")
    require(child["paid_ceiling_usd"] <= 20 and child["stage_spent_usd"] >= 0,
            "Original twenty-dollar cycle envelope exceeded")
    require(master["execution_hardware"] == pm["execution_hardware"], "This continuation requires the same MIG/runtime")
    pins = master["source_sha256"]
    require(all(pins.get(name) == value for name, value in pm["source_sha256"].items()), "Original B2 source pins changed")
    require(pins.get("scripts/run_babylm_scale_epoch_continuation_v0.py") == sha(__file__), "Continuation controller is not pinned")
    for name, value in pins.items(): require(sha(resolve(name)) == value, "Source SHA mismatch: " + name)
    manifest_path = resolve(child["train_manifest"])
    manifest = load(manifest_path)
    index = np.load(manifest_path.parent / "windows.u64.npy", mmap_mode="r", allow_pickle=False)
    order = np.random.default_rng(np.random.SeedSequence([child["data_order_seed"], 0])).permutation(len(index))
    return {"kind": "validation_only_no_model_calls", "child_protocol": child, "child_protocol_sha256": canonical_sha(child),
            "parent_counts": parent["summary"]["counts"], "parent_eval_counts": parent["summary"]["eval_counts"],
            "parent_elapsed_seconds": elapsed, "parent_log_bytes": parent["log_bytes"], "parent_log_sha256": parent["log_sha256"],
            "expected_epoch_counts": epoch_plan["expected_counts"], "window_ids": [int(i) for i in order],
            "expected_panel_updates": [0, 250, 500, 750, 1000, 1250, 1413],
            "expected_new_updates": 913, "expected_new_training_forwards": 14598,
            "expected_new_training_backwards": 14598, "expected_new_eval_forwards": 192,
            "expected_tail_updates": [1314, 1413], "source": str(source)}


def command(protocol_path, run_dir):
    return [sys.executable, "-u", str(ROOT / "scripts/run_babylm_de_v0.py"), "--protocol", str(protocol_path),
            "--mode", "sparse", "--output-dir", str(run_dir), "--resume-checkpoint", str(run_dir / "checkpoint.pt")]


def audit_continuation(run_dir, plan, branch, master, master_sha):
    """Final ledger check permits exactly the explicitly authorized u500 resume."""
    run_dir = Path(run_dir)
    raw = (run_dir / "events.jsonl").read_bytes()
    n = branch["historical_log_prefix_bytes"]
    require(hashlib.sha256(raw[:n]).hexdigest() == branch["historical_log_prefix_sha256"], "Historical log bytes changed")
    require(raw.endswith(b"\n"), "Terminal log is incomplete")
    events = [json.loads(line) for line in raw.splitlines()]
    require([e["event_id"] for e in events] == list(range(1, len(events) + 1)), "Continuation event sequence differs")
    new_events = [json.loads(line) for line in raw[n:].splitlines()]
    require(new_events and new_events[0]["type"] == "resume", "First branch event must be the explicit resume")
    require(sum(e["type"] == "resume" for e in events) == 1, "Unexpected additional resume")
    require(not any(e["type"] in {"failure", "run_error", "run_resume", "consumed_without_update", "eval_skipped_budget"} for e in events),
            "Failed, duplicate, or budget-skipped continuation evidence")
    require(new_events[0]["checkpoint_sha256"] == branch["new_checkpoint_sha256"]
            and new_events[0]["counts"] == branch["initial_counts"] and new_events[0]["cursor"] == branch["initial_cursor"],
            "Resume did not use the registered administrative branch")
    protocol, summary = load(run_dir / "protocol.json"), load(run_dir / "summary.json")
    require(protocol == plan["child_protocol"], "Executed continuation protocol differs")
    require(summary["status"] == "epoch_complete" and summary["cursor"] == {"epoch": 1, "position": 0}, "Scientific epoch is incomplete")
    require(summary["protocol_sha256"] == canonical_sha(protocol), "Final protocol SHA differs")
    updates = [e for e in events if e["type"] == "update"]
    require(len(updates) == 1413 and [e["counts"]["updates"] for e in new_events if e["type"] == "update"] == list(range(501, 1414)),
            "New training must be exactly updates 501 through 1413")
    seen, totals = [], {k: 0 for k in ("input_tokens", "loss_tokens", "word_exposures")}
    for i, event in enumerate(updates, 1):
        require(event["counts"]["updates"] == event["counts"]["scientific_updates"] == i and event["counts"]["engineering_updates"] == 0,
                "Scientific update ledger differs")
        require(len(event["windows"]) == (6 if i == 1413 else 16), "Window accumulation differs")
        for row in event["windows"]:
            require(row["epoch"] == 0 and row["loss_tokens"] == row["input_tokens"] - 1 and
                    all(row[k] is True for k in ("forward_started", "forward_completed", "backward_started", "backward_completed")),
                    "Incomplete or repeated-epoch training window")
            seen.append(row["window_index"])
            for key in totals: totals[key] += row[key]
        require(all(event["counts"][key] == value for key, value in totals.items()), "Raw cumulative accounting differs")
        require(event["loss_tokens_this_update"] == sum(w["loss_tokens"] for w in event["windows"]), "LM target denominator differs")
        require(math.isfinite(event["loss_token_weighted_ce"]) and math.isfinite(event["loss_token_weighted_aux"]), "Nonfinite LM/auxiliary loss")
        validate_learning_rate(event, protocol, event["counts"]["word_exposures"])
    require(seen == plan["window_ids"] and len(seen) == len(set(seen)) == 22598, "Epoch order/coverage differs")
    for key, value in plan["expected_epoch_counts"].items():
        if key != "final_update_windows": require(summary["counts"][key] == value, "Final frozen count differs: " + key)
    require(summary["counts"] == updates[-1]["counts"] == events[-1]["counts"] and events[-1]["type"] == "run_stop"
            and events[-1]["status"] == "epoch_complete", "Final terminal counters differ")
    panels = [e for e in events if e["type"] == "evaluation"]
    require([e["counts"]["updates"] for e in panels] == plan["expected_panel_updates"], "Panel cadence/replay differs")
    eval_sum = {k: 0 for k in ("forward_calls", "input_tokens", "loss_tokens", "word_exposures")}
    for event in panels:
        metrics = event["metrics"]
        require(metrics["window_indices"] == protocol["eval_window_indices"] and metrics["total"]["forward_calls"] == 48,
                "Panel window set differs")
        require(math.isfinite(metrics["total"]["nll"]) and metrics["optimizer_updates"] == 0
                and metrics["grad_enabled_during_model_forward"] is False and metrics["auxiliary_loss_included_in_nll"] is False,
                "Panel is not finite pure LM no-grad evaluation")
        for key in eval_sum: eval_sum[key] += metrics["total"][key]
    require(summary["eval_counts"] == dict(eval_sum, evaluations=7) and eval_sum["forward_calls"] == 336, "Evaluation ledger differs")
    require(sha(run_dir / "checkpoint.pt") == summary["checkpoint_sha256"], "Full checkpoint SHA differs")
    finals = [x for x in summary["snapshot_state"]["final_receipts"] if x["stop_reason"] == "epoch_complete" and x["point"]["updates"] == 1413]
    require(len(finals) == 1 and finals[0]["protocol_completion_boundary"] is True, "Unique complete-epoch model receipt required")
    final = finals[0]
    require(sha(run_dir / final["path"]) == final["sha256"] and sha(run_dir / final["model_path"]) == final["model_sha256"],
            "Final model-only checkpoint/receipt SHA differs")
    incremental = {key: summary["counts"][key] - branch["initial_counts"][key] for key in summary["counts"]}
    incremental_eval = {key: summary["eval_counts"][key] - branch["initial_eval_counts"][key] for key in summary["eval_counts"]}
    require(incremental["updates"] == 913 and incremental["forward_calls"] == incremental["backward_calls"] == 14598
            and incremental_eval["forward_calls"] == 192, "Incremental physical work differs")
    tail = updates[1313:1413]
    denominator = sum(e["loss_tokens_this_update"] for e in tail)
    nll = math.fsum(e["loss_token_weighted_ce"] * e["loss_tokens_this_update"] for e in tail) / denominator
    return {"status": "complete_epoch_continuation_audited", "master_protocol_sha256": master_sha,
            "protocol_sha256": canonical_sha(protocol), "child_protocol_sha256": canonical_sha(protocol),
            "source_sha256": master["source_sha256"], "counts": summary["counts"], "eval_counts": summary["eval_counts"],
            "incremental_counts": incremental, "incremental_eval_counts": incremental_eval,
            "historical_prefix_bytes_unchanged": True, "authorized_resume_count": 1,
            "tail": {"first_update": 1314, "last_update": 1413, "loss_tokens": denominator,
                     "token_weighted_lm_nll": nll, "token_weighted_lm_ppl": math.exp(nll),
                     "arithmetic_mean_step_lm_ppl": math.fsum(math.exp(e["loss_token_weighted_ce"]) for e in tail) / 100,
                     "scope": "online changing-model training loss, not held-out evaluation"},
            "panel_updates": plan["expected_panel_updates"], "panel_metrics": [e["metrics"] for e in panels],
            "final_model_checkpoint_path": str((run_dir / final["model_path"]).relative_to(ROOT)).replace("\\", "/"),
            "final_model_checkpoint_sha256": final["model_sha256"], "final_model_receipt_path": str(run_dir / final["path"]),
            "final_full_checkpoint_sha256": summary["checkpoint_sha256"], "analysis_model_forward_calls": 0,
            "analysis_backward_calls": 0, "analysis_optimizer_updates": 0}


def execute(master_path, digest):
    started = time.monotonic()
    require(os.name == "posix", "Scientific continuation execution requires Linux")
    master_path = resolve(master_path)
    require(sha(master_path) == digest, "Explicit frozen master SHA required")
    master = load(master_path)
    require(master.get("launch_allowed") is True, "Continuation launch not frozen")
    plan = validate(master)
    out = resolve(master["output_dir"])
    require(out.is_relative_to(ROOT / "results") and not out.exists(), "New results directory required")
    import fcntl
    lock_path = "/tmp/babylm-one-epoch-" + master["execution_hardware"]["mig_uuid"] + ".lock"
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = prefix.migration_gate(master, os.environ.get("RUNPOD_POD_ID"))
        out.mkdir(parents=True, exist_ok=False)
        state = {"status": "preparing_administrative_branch", "pid": os.getpid(), "started_utc": utc(),
                 "protocol_sha256": digest, "gate_receipt_sha256": master["gate_receipt_sha256"], "pod_id": gate["pod_id"],
                 "execution_hardware": master["execution_hardware"], "lock_path": lock_path,
                 "hard_timeout_seconds_including_preparation": master["hard_timeout_seconds"], "no_retry_or_new_experiment": True}
        write(out / "stage.json", state)
        previous_handler = signal.getsignal(signal.SIGTERM)
        def stopped(signum, frame): raise InterruptedError("Continuation controller received stop signal")
        signal.signal(signal.SIGTERM, stopped)
        try:
            child_path = out / "child-protocol.json"
            write(child_path, plan["child_protocol"])
            branch = prepare_branch(plan["source"], out / "run", plan["child_protocol"], master["parent_file_sha256"])
            remaining = master["hard_timeout_seconds"] - (time.monotonic() - started)
            require(remaining > 30, "Preparation exhausted the training hard deadline")
            prefix.supervise(command(child_path, out / "run"), out, remaining, state)
            verify_files(plan["source"], master["parent_file_sha256"])
            audit = audit_continuation(out / "run", plan, branch, master, digest)
            write(out / "continuation-audit.json", audit)
            state.update(status="complete_epoch_continuation_audited", counts=audit["counts"], eval_counts=audit["eval_counts"],
                         incremental_counts=audit["incremental_counts"], incremental_eval_counts=audit["incremental_eval_counts"])
            return 0
        except BaseException as error:
            state.update(status="failed_or_incomplete", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            state.update(finished_utc=utc(), elapsed_wall_seconds=time.monotonic() - started)
            state["estimated_new_worker_cost_usd"] = state["elapsed_wall_seconds"] / 3600 * plan["child_protocol"]["hourly_rate_usd"]
            state["cost_scope"] = "New wall estimate only; inherited engine elapsed/cost includes B2 and must not be summed again"
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
        require(bool(args.protocol_sha256), "Explicit --protocol-sha256 required")
        return execute(args.protocol, args.protocol_sha256)
    plan = validate(load(resolve(args.protocol)))
    print(json.dumps({k: v for k, v in plan.items() if k not in {"child_protocol", "window_ids"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
