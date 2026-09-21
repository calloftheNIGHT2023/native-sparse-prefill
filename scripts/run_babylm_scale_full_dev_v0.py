"""Evaluate the single prespecified scale-only final-epoch checkpoint on full dev.

This new wrapper never trains, resumes, chooses a checkpoint by score, or edits
the frozen evaluator. A finite outer controller supplies the one-hour process-
group timeout; this wrapper holds the shared MIG lock itself. Default CLI is
source/template validation only.
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
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/run_babylm_scale_full_dev_v0.py"
ENGINE = "scripts/run_babylm_checkpoint_eval_v0.py"
BASE_TEMPLATE = "configs/babylm-optimization-stage-a-20260920-v0/full-dev-sparse.json"
BASE_TEMPLATE_SHA = "7bc1fd79a2007299aaaa828f0e102fc7f556f6fa33b94236ee349a68d8b66802"
DEV_MANIFEST = "data/babylm-dev-windows-v0/manifest.json"
DEV_MANIFEST_SHA = "baa53c08d1c26e6dda2f238d2221a1d2cfef4c754ec46327765987f423519619"
FINAL_STEM = "model-u00001413-w000010001709-i000016325414-l000016302816"
TRAIN_COUNTS = {"updates": 1413, "scientific_updates": 1413, "engineering_updates": 0,
                "windows": 22598, "forward_calls": 22598, "backward_calls": 22598,
                "word_exposures": 10001709, "input_tokens": 16325414, "loss_tokens": 16302816}
NEW_TRAIN_COUNTS = {"updates": 913, "scientific_updates": 913, "engineering_updates": 0,
                    "windows": 14598, "forward_calls": 14598, "backward_calls": 14598}
DEV_COUNTS = {"windows": 18792, "forward_calls": 18792, "input_tokens": 17437534,
              "loss_tokens": 17418742, "word_exposures": 10420962}
SCALE = 1.0 / math.sqrt(128)
ALLOWED_CHILD_CHANGES = {"checkpoint_path", "checkpoint_sha256", "checkpoint_protocol_sha256",
                         "model_config", "output_dir", "max_wall_seconds"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value, exclusive=False):
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def validate_template(template):
    require(template["schema_version"] == 1 and template["scope"] == "scientific_evaluation", "Frozen evaluation scope required")
    require(template["mode"] == "sparse" and template["device"] == "cuda" and template["dtype"] == "float32", "Original sparse CUDA FP32 evaluator required")
    require(template["window_indices"] is None and template["position_diagnostics"] is True,
            "Full development set with original position scoring required")
    require(template["enable_routing_diagnostics"] is False and template["routing_policy"] == "learned", "No routing intervention or extra model forwards")
    require(template["dev_manifest"] == DEV_MANIFEST and template["dev_manifest_sha256"] == DEV_MANIFEST_SHA,
            "Development dataset replacement is forbidden")
    require(template["model_config"]["index_score_scale"] == 1.0 and template["model_config"]["index_head_dim"] == 128,
            "Expected original unit-scale baseline template")


def derive_child(template, master, checkpoint_path, checkpoint_sha, training_protocol_sha):
    validate_template(template)
    require(master["index_score_scale"] == SCALE, "Only frozen 1/sqrt(128) score scale is allowed")
    child = copy.deepcopy(template)
    child["model_config"]["index_score_scale"] = SCALE
    child.update(checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha,
                 checkpoint_protocol_sha256=training_protocol_sha,
                 output_dir=str(Path(master["output_dir"]) / "evaluation"),
                 max_wall_seconds=3500)
    changed = {key for key in set(template) | set(child) if template.get(key) != child.get(key)}
    require(changed <= ALLOWED_CHILD_CHANGES, "Unexpected evaluation protocol mutation")
    nested = {key for key in set(template["model_config"]) | set(child["model_config"])
              if template["model_config"].get(key) != child["model_config"].get(key)}
    require(nested == {"index_score_scale"}, "Model changes beyond score scale are forbidden")
    return child, {key: {"before": template.get(key), "after": child.get(key)} for key in sorted(changed)}


def validate(master):
    require(master.get("schema_version") == 1, "Unsupported full-dev master schema")
    require(master["base_template"] == BASE_TEMPLATE and master["base_template_sha256"] == BASE_TEMPLATE_SHA,
            "Original Stage A full-dev sparse template must remain byte-identical")
    require(sha(resolve(BASE_TEMPLATE)) == BASE_TEMPLATE_SHA, "Base full-dev template SHA differs")
    template = load(resolve(BASE_TEMPLATE))
    validate_template(template)
    require(master["soft_timeout_seconds"] == 3500 and master["hard_timeout_seconds"] == 3600,
            "This full-dev stage has 3500-second soft and 3600-second hard limits")
    require(0 < master["hourly_rate_usd"] <= 2 and master["hourly_rate_usd"] <= master["stage_cost_cap_usd"] <= 2,
            "One-hour full-dev bound must fit its frozen cost ceiling")
    require(master["index_score_scale"] == SCALE, "Scale differs from frozen intervention")
    hardware = master["execution_hardware"]
    require(hardware["mig_uuid"].startswith("MIG-") and hardware["physical_uuid"].startswith("GPU-"), "Explicit hardware identities required")
    uuid.UUID(hardware["mig_uuid"][4:])
    uuid.UUID(hardware["physical_uuid"][4:])
    require(hardware["mig_profile"] == "2g.48gb" and isinstance(hardware["driver_version"], str), "Frozen 48GB MIG execution hardware required")
    pins = master["source_sha256"]
    require(pins.get(SELF) == sha(Path(__file__)), "New wrapper source must be pinned")
    require(all(pins.get(name) == digest for name, digest in template["expected_source_hashes"].items()),
            "Every original evaluator source pin must be retained")
    for name, digest in pins.items():
        path = resolve(name)
        require(path.is_relative_to(ROOT) and sha(path) == digest, "Source pin differs: " + name)
    train_master = resolve(master["continuation_master"])
    require(train_master.is_relative_to(ROOT / "configs") and sha(train_master) == master["continuation_master_sha256"],
            "Frozen continuation master SHA differs")
    run_dir = resolve(master["training_run_dir"])
    out = resolve(master["output_dir"])
    require(run_dir.is_relative_to(ROOT / "results") and out.is_relative_to(ROOT / "results")
            and not out.is_relative_to(run_dir) and not run_dir.is_relative_to(out), "Separate bounded training/evaluation paths required")
    require(resolve(master["continuation_audit_path"]) == run_dir.parent / "continuation-audit.json",
            "Only the predetermined continuation audit is allowed")
    require(sha(resolve(DEV_MANIFEST)) == DEV_MANIFEST_SHA, "Frozen development manifest SHA differs")
    manifest = load(resolve(DEV_MANIFEST))
    require(manifest["total_windows"] == DEV_COUNTS["windows"] and manifest["input_tokens_per_pass"] == DEV_COUNTS["input_tokens"]
            and manifest["next_token_loss_positions_per_pass"] == DEV_COUNTS["loss_tokens"]
            and manifest["word_exposures_per_pass"] == DEV_COUNTS["word_exposures"] and len(manifest["source_summaries"]) == 6,
            "Full development-set ledger differs")
    return template, manifest


def validate_completion(summary, audit, final_receipt, train_protocol, master, template, checkpoint_path, checkpoint_sha):
    require(summary.get("status") == "epoch_complete" and summary.get("mode") == "sparse", "Only completed sparse epoch may enter full dev")
    require(audit.get("status") == "complete_epoch_continuation_audited", "Continuation count/source audit did not pass")
    require(audit.get("master_protocol_sha256") == master["continuation_master_sha256"], "Continuation audit belongs to another master")
    require(summary["counts"] == audit["counts"] == final_receipt["counts"], "Summary/audit/final receipt counts disagree")
    for key, value in TRAIN_COUNTS.items():
        require(summary["counts"][key] == value, "Unexpected completed training count: " + key)
    for key, value in NEW_TRAIN_COUNTS.items():
        require(audit["incremental_counts"][key] == value, "Unexpected continuation count: " + key)
    require(summary["eval_counts"]["forward_calls"] == audit["eval_counts"]["forward_calls"] == 336,
            "Original cumulative panel-forward accounting differs")
    require(final_receipt["stop_reason"] == "epoch_complete" and final_receipt["protocol_completion_boundary"] is True,
            "Final receipt is not the prespecified epoch boundary")
    require(final_receipt["point"] == {key: TRAIN_COUNTS[key] for key in ("updates", "word_exposures", "input_tokens", "loss_tokens")},
            "Final model point differs; score-based checkpoint selection is forbidden")
    require(final_receipt["model_path"] == "snapshots/" + FINAL_STEM + ".pt", "Wrong final model path")
    require(final_receipt["model_sha256"] == audit["final_model_checkpoint_sha256"] == checkpoint_sha,
            "Final model SHA provenance differs")
    require(resolve(audit["final_model_checkpoint_path"]) == resolve(checkpoint_path), "Continuation audit names another checkpoint")
    training_sha = canonical_sha(train_protocol)
    require(summary["protocol_sha256"] == audit["protocol_sha256"] == final_receipt["protocol_sha256"] == training_sha,
            "Training protocol provenance differs")
    if "training_protocol_sha256" in master:
        require(training_sha == master["training_protocol_sha256"], "Pre-frozen training protocol SHA differs")
    expected_model = copy.deepcopy(template["model_config"])
    expected_model["index_score_scale"] = SCALE
    require(train_protocol["model_config"] == expected_model, "Training architecture differs beyond score scale")
    require(train_protocol["backbone_seed"] == template["backbone_seed"] and train_protocol["indexer_seed"] == template["indexer_seed"],
            "Training seeds differ")
    require(train_protocol["max_epochs"] == 1 and train_protocol["max_updates"] == 1413, "Training horizon differs")
    require(summary["source_hashes"] == final_receipt["source_hashes"], "Training source receipts disagree")
    return training_sha


def bind_checkpoint(master, template):
    run_dir = resolve(master["training_run_dir"])
    paths = {"training_summary": run_dir / "summary.json", "continuation_audit": resolve(master["continuation_audit_path"]),
             "training_protocol": run_dir / "protocol.json", "final_receipt": run_dir / "snapshots" / (FINAL_STEM + "-final-epoch_complete.json"),
             "checkpoint": run_dir / "snapshots" / (FINAL_STEM + ".pt")}
    hashes = {name: sha(path) for name, path in paths.items()}
    training_sha = validate_completion(load(paths["training_summary"]), load(paths["continuation_audit"]), load(paths["final_receipt"]),
        load(paths["training_protocol"]), master, template, str(paths["checkpoint"]), hashes["checkpoint"])
    child, changes = derive_child(template, master, str(paths["checkpoint"]), hashes["checkpoint"], training_sha)
    receipt = {"status": "prespecified_final_1413_bound_after_training_audit", "bound_utc": utc(),
        "base_template": BASE_TEMPLATE, "base_template_sha256": BASE_TEMPLATE_SHA,
        "continuation_master": master["continuation_master"], "continuation_master_sha256": master["continuation_master_sha256"],
        "bound_sources": {name: {"path": str(paths[name]), "sha256": value} for name, value in hashes.items()},
        "checkpoint_hash_source": "Computed from the only prespecified final1413 model, then matched to final receipt and continuation audit",
        "training_protocol_canonical_sha256": training_sha, "derived_evaluation_protocol_canonical_sha256": canonical_sha(child),
        "changed_fields": changes, "scientific_change": {"model_config.index_score_scale": {"before": 1., "after": SCALE}},
        "administrative_binding_fields": ["checkpoint_path", "checkpoint_sha256", "checkpoint_protocol_sha256", "output_dir", "max_wall_seconds"],
        "checkpoint_selection": "No score read for selection; fixed final update1413 only; no fallback checkpoint",
        "data_and_scoring_unchanged": True, "new_training_updates": 0, "new_backward_calls": 0}
    return child, receipt


def audit_evaluation(summary, rows, manifest, child):
    require(summary.get("status") == "evaluation_complete" and summary.get("partial_metrics_only") is False,
            "Full development evaluation is incomplete")
    require(summary.get("optimizer_updates") == 0 and summary.get("backward_calls") == 0, "Evaluation performed training work")
    require(summary["counts"]["optimizer_updates"] == summary["counts"]["backward_calls"] == 0, "Unexpected evaluator counters")
    require(summary["requested_windows"] == DEV_COUNTS["windows"] and summary["full_manifest_requested"] is True,
            "Only the complete frozen development manifest is allowed")
    expected_ids = list(range(DEV_COUNTS["windows"]))
    require(summary["window_indices"] == summary["observed_window_indices"] == expected_ids
            and [row["window_index"] for row in rows] == expected_ids, "Development window order/membership differs")
    require(summary["checkpoint"]["sha256"] == child["checkpoint_sha256"] and summary["checkpoint"]["optimizer_state_loaded"] is False,
            "Evaluated checkpoint provenance differs")
    for key, value in DEV_COUNTS.items():
        require(summary["total"][key] == value, "Full development count differs: " + key)
    for key in ("input_tokens", "loss_tokens", "word_exposures", "forward_calls"):
        require(sum(row[key] for row in rows) == DEV_COUNTS[key], "Raw evaluation rows disagree: " + key)
    require(all(row["grad_enabled"] is False and row["model_training"] is False and row["routing_policy"] == "learned" for row in rows),
            "Evaluation mode or routing changed")
    nll = math.fsum(row["nll_sum"] for row in rows) / DEV_COUNTS["loss_tokens"]
    require(math.isfinite(nll) and math.isclose(nll, summary["total"]["nll"], rel_tol=0, abs_tol=1e-10), "Raw loss aggregation disagrees")
    require(math.isfinite(summary["total"]["ppl"]) and math.isclose(math.exp(nll), summary["total"]["ppl"], rel_tol=1e-10), "PPL aggregation disagrees")
    expected_sources = {source["source"]: source for source in manifest["source_summaries"]}
    require(set(summary["per_source"]) == set(expected_sources), "Expected exactly the original six sources")
    for name, source in expected_sources.items():
        observed = summary["per_source"][name]
        for metric, ledger_key in (("windows", "windows"), ("input_tokens", "input_tokens"),
                                  ("loss_tokens", "next_token_loss_positions"), ("word_exposures", "window_word_exposures")):
            require(observed[metric] == source[ledger_key], "Per-source ledger differs: " + name + "/" + metric)
    diagnostics = summary["position_diagnostics"]
    require(diagnostics["protocol"] == "babylm-dev-position-v0" and diagnostics["extra_model_forwards"] == 0
            and set(diagnostics["per_source_query_history_bins"]) == set(expected_sources), "Position scoring protocol differs")
    return {"status": "complete_full_dev_audited", "counts": dict(DEV_COUNTS), "nll": nll, "ppl": math.exp(nll),
            "source_count": 6, "checkpoint_sha256": child["checkpoint_sha256"], "training_updates": 0,
            "backward_calls": 0, "data_scoring_unchanged": True,
            "scope": "Full development evidence; not a held-out confirmation or sparse-speed measurement"}


def run_under_lock(master, master_sha, lock_path):
    require(master.get("launch_allowed") is True, "Full-dev launch is not authorized in this frozen master")
    template, manifest = validate(master)
    out = resolve(master["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "master.json", master, exclusive=True)
    started = time.monotonic()
    state = {"status": "binding_completed_epoch", "pid": os.getpid(), "started_utc": utc(), "master_protocol_sha256": master_sha,
             "new_training_updates": 0, "new_backward_calls": 0, "automatic_monitoring_resumed": False,
             "soft_timeout_seconds": 3500, "external_hard_timeout_seconds": 3600, "mig_lock_path": lock_path}
    child = None
    def stopped(signum, frame):
        raise InterruptedError("Full-dev wrapper received signal " + str(signum))
    previous = signal.signal(signal.SIGTERM, stopped)
    write(out / "stage.json", state)
    try:
        hardware = master["execution_hardware"]
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True, timeout=20).stdout
        require(re.findall(r"\(UUID: (GPU-[0-9a-fA-F-]+)\)", listing) == [hardware["physical_uuid"]], "Visible physical GPU differs")
        require(re.findall(r"MIG\s+(\S+)\s+Device\s+\d+:\s*\(UUID:\s*(MIG-[0-9a-fA-F-]+)\)", listing)
                == [(hardware["mig_profile"], hardware["mig_uuid"])], "Visible MIG instance differs")
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True, timeout=20).stdout.strip()
        require(driver == hardware["driver_version"], "GPU driver differs from frozen execution hardware")
        protocol, receipt = bind_checkpoint(master, template)
        write(out / "source-binding-receipt.json", receipt, exclusive=True)
        protocol_path = out / "evaluation-protocol.json"
        write(protocol_path, protocol, exclusive=True)
        state.update(status="evaluating", evaluation_protocol_sha256=sha(protocol_path))
        remaining = 3500 - (time.monotonic() - started)
        require(remaining > 60, "No time remaining after final-checkpoint validation")
        with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr:
            child = subprocess.Popen([sys.executable, "-u", str(resolve(ENGINE)), "--protocol", str(protocol_path)],
                                     cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=False)
            state["child_pid"] = child.pid
            write(out / "stage.json", state)
            code = child.wait(timeout=remaining)
        require(code == 0, "Original full-dev evaluator failed; preserve partial evidence, no automatic retry")
        evaluation_dir = resolve(protocol["output_dir"])
        summary = load(evaluation_dir / "summary.json")
        rows_path = evaluation_dir / "windows.jsonl"
        require(sha(rows_path) == summary["windows_jsonl_sha256"], "Raw evaluation rows SHA differs")
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
        audit = audit_evaluation(summary, rows, manifest, protocol)
        require(time.monotonic() - started < 3500, "Full-dev soft deadline exceeded during audit")
        validate(master)
        audit.update(master_protocol_sha256=master_sha, evaluation_protocol_sha256=sha(protocol_path),
                     source_binding_receipt_sha256=sha(out / "source-binding-receipt.json"),
                     summary_sha256=sha(evaluation_dir / "summary.json"), windows_jsonl_sha256=sha(rows_path),
                     completed_utc=utc())
        write(out / "full-dev-audit.json", audit, exclusive=True)
        state.update(status="complete_full_dev_audited", evaluation_forward_calls=DEV_COUNTS["forward_calls"],
                     nll=audit["nll"], ppl=audit["ppl"])
        return 0
    except BaseException as error:
        state.update(status="failed_or_incomplete", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        return 2
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=20)
        if child is not None:
            state["child_returncode"] = child.poll()
        state.update(finished_utc=utc(), elapsed_wall_seconds=time.monotonic() - started)
        state["estimated_stage_cost_usd"] = state["elapsed_wall_seconds"] / 3600 * master["hourly_rate_usd"]
        state["cost_scope"] = "This wrapper wall time only; no overlap with training stage, not invoice"
        write(out / "stage.json", state)
        signal.signal(signal.SIGTERM, previous)


def run(master, master_sha):
    require(os.name == "posix", "The finite full-dev worker requires Linux")
    validate(master)
    import fcntl
    lock_path = "/tmp/babylm-one-epoch-" + master["execution_hardware"]["mig_uuid"] + ".lock"
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run_under_lock(master, master_sha, lock_path)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(sha(args.protocol) == args.protocol_sha256, "Full-dev master SHA differs")
    master = load(args.protocol)
    validate(master)
    if not args.execute:
        print(json.dumps({"status": "validated_no_model_calls", "fixed_final_update": 1413,
                          "development_counts": DEV_COUNTS, "soft_timeout_seconds": 3500, "hard_timeout_seconds": 3600}))
        return 0
    return run(master, args.protocol_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
