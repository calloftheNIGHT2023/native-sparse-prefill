"""One fresh fixed-local W epoch then one full-dev evaluation, no retry.

This administrative process owns the shared MIG flock. Both children remain
inside its externally bounded process group. The evaluation checkpoint hash
is bound only after the unique final1413 training audit succeeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/run_babylm_local_queue_v0.py"
TRAIN_SCRIPT = "scripts/run_babylm_local_train_v0.py"
EVAL_SCRIPT = "scripts/run_babylm_local_eval_v0.py"
FINAL_STEM = "model-u00001413-w000010001709-i000016325414-l000016302816"
COUNTS = dict(updates=1413, windows=22598, forward_calls=22598,
              backward_calls=22598, input_tokens=16325414,
              loss_tokens=16302816, word_exposures=10001709)


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def resolve(value):
    p = (ROOT / value).resolve()
    require(p.is_relative_to(ROOT), "Path outside project")
    return p


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def validate(master):
    require(master["schema_version"] == 1 and master["launch_allowed"] is True, "Unfrozen queue")
    require(master["condition"] == "W_fixed_local", "Wrong scientific condition")
    require(master["hard_timeout_seconds"] == 10800, "Three-hour queue limit required")
    require(master["train_timeout_seconds"] == 6900 and master["eval_timeout_seconds"] == 3300,
            "Stage bounds changed")
    require(master["execution_cap_usd"] == 4.2 and master["hourly_rate_usd"] == 1.4,
            "Frozen execution cost cap changed")
    require(master["cycle_cap_usd"] == 20 and master["whole_batch_cap_usd"] == 6, "Batch/cycle cap changed")
    require(master["known_cycle_usd_before_queue"] + master["execution_cap_usd"] <= master["cycle_cap_usd"],
            "Insufficient recorded cycle allocation")
    require(master["known_preparation_usd"] + master["execution_cap_usd"] <= master["whole_batch_cap_usd"],
            "Preparation exhausted this batch allocation")
    pins = master["source_sha256"]
    require(all(s in pins for s in (SELF, TRAIN_SCRIPT, EVAL_SCRIPT)), "Missing queue/worker source pin")
    for name, digest in pins.items():
        require(sha(resolve(name)) == digest, "Source changed: " + name)
    for name in ("training_master", "evaluation_template", "gate_receipt", "authorization"):
        require(sha(resolve(master[name])) == master[name + "_sha256"], "Binding changed: " + name)
    gate = load(resolve(master["gate_receipt"]))
    require(gate["status"] == "passed_local_gpu_and_throughput_gate", "Unpassed new-local gate")
    require(gate["execution_hardware"] == master["execution_hardware"], "Gate hardware differs")
    require(gate["scientific_weights_reused"] is False, "Engineering weights cannot initialize science")
    out = resolve(master["output_dir"])
    require(out.parent == ROOT / "results", "Dedicated results directory required")
    train = load(resolve(master["training_master"]))
    require(train["condition"] == "W_fixed_local" and train.get("purpose", "scientific") == "scientific",
            "Queue must run scientific W")
    require(resolve(train["output_dir"]) == out / "train", "Training output mismatch")
    evaluation = load(resolve(master["evaluation_template"]))
    require(evaluation["condition"] == "W_fixed_local" and evaluation["engine_compat_mode"] == "dense",
            "W evaluation condition must be explicit")
    require(evaluation["window_indices"] is None and evaluation["position_diagnostics"] is True,
            "Only full fixed dev is allowed")
    require(resolve(evaluation["output_dir"]) == out / "full-dev", "Evaluation output mismatch")
    return out, train, evaluation


def bind_evaluation(master, out, template):
    run = out / "train" / "run"
    summary = load(run / "summary.json")
    audit_path = out / "train" / "local-training-audit.json"
    audit = load(audit_path)
    require(summary["status"] == "epoch_complete", "Training did not complete original epoch")
    require(audit.get("condition") == "W_fixed_local", "Missing W training audit")
    require(audit.get("status") == "complete_local_epoch_audited", "Failed W training audit")
    for key, value in COUNTS.items():
        require(summary["counts"][key] == value, "Incorrect final count: " + key)
    require(summary["eval_counts"]["forward_calls"] == 336, "Incorrect original panel count")
    protocol = load(run / "protocol.json")
    require(protocol["scientific_condition"] == "W_fixed_local", "W condition missing from saved protocol")
    checkpoint = run / "snapshots" / (FINAL_STEM + ".pt")
    final_receipt = load(run / "snapshots" / (FINAL_STEM + "-final-epoch_complete.json"))
    checkpoint_sha = sha(checkpoint)
    require(final_receipt["model_sha256"] == checkpoint_sha, "Final model bytes differ")
    require(final_receipt["stop_reason"] == "epoch_complete", "Wrong final snapshot reason")
    require(final_receipt["point"]["updates"] == 1413, "Score-based checkpoint selection forbidden")
    child = dict(template)
    child.update(checkpoint_path=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=checkpoint_sha,
                 checkpoint_protocol_sha256=summary["protocol_sha256"],
                 training_audit_path=str(audit_path.relative_to(ROOT)), training_audit_sha256=sha(audit_path))
    child_path = out / "bound-full-dev-protocol.json"
    require(not child_path.exists(), "Refusing to overwrite evaluation binding")
    write(child_path, child)
    write(out / "final-checkpoint-binding.json", dict(condition="W_fixed_local", utc=utc(),
        checkpoint_path=child["checkpoint_path"], checkpoint_sha256=checkpoint_sha,
        training_protocol_sha256=summary["protocol_sha256"], training_audit_sha256=sha(audit_path),
        final_receipt_sha256=sha(run / "snapshots" / (FINAL_STEM + "-final-epoch_complete.json")),
        checkpoint_selection="prespecified final1413 only, no score selection", model_calls=0))
    return child_path, sha(child_path)


def execute(path, digest):
    import fcntl
    require(os.name == "posix", "Linux cloud required")
    require(sha(path) == digest, "Master changed")
    master = load(path)
    out, train, template = validate(master)
    require(os.environ.get("RUNPOD_POD_ID") == master["pod_id"], "Wrong Pod")
    require(os.getppid() == os.getpgrp(), "Require dedicated timeout parent/group")
    listing = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=15)
    hw = master["execution_hardware"]
    require(hw["physical_uuid"] in listing and hw["mig_uuid"] in listing, "GPU changed since preflight")
    lock = open("/tmp/babylm-one-epoch-" + hw["mig_uuid"] + ".lock", "a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not out.exists(), "No overwrite/retry of W science")
    out.mkdir(parents=True)
    write(out / "master.json", master)
    started = time.monotonic()
    state = dict(status="running", condition="W_fixed_local", pid=os.getpid(), pgid=os.getpgrp(),
                 protocol_sha256=digest, started_utc=utc(), stages=[], no_automatic_retry=True,
                 queue_forward_calls=0, queue_backward_calls=0, queue_optimizer_updates=0)
    abnormal = False
    def interrupted(signum, frame):
        raise InterruptedError("Bounded W queue terminated")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for name, script, limit in [("train", TRAIN_SCRIPT, 6900), ("full_dev", EVAL_SCRIPT, 3300)]:
            validate(master)
            if name == "train":
                protocol_path = resolve(master["training_master"])
                protocol_sha = master["training_master_sha256"]
            else:
                protocol_path, protocol_sha = bind_evaluation(master, out, template)
            remaining = min(limit, 10800 - 30 - (time.monotonic() - started))
            require(remaining > 60, "Queue remaining time exhausted")
            row = dict(name=name, started_utc=utc(), status="starting")
            state["stages"].append(row)
            command = [sys.executable, "-u", str(resolve(script)), "--protocol", str(protocol_path),
                       "--protocol-sha256", protocol_sha, "--execute"]
            with (out / (name + ".stdout.log")).open("xb") as stdout, (out / (name + ".stderr.log")).open("xb") as stderr:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=False)
                require(os.getpgid(child.pid) == os.getpgrp(), "Child escaped process group")
                row.update(status="running", pid=child.pid, pgid=os.getpgrp())
                state.update(active_stage=name, child_pid=child.pid)
                write(out / "stage.json", state)
                code = child.wait(timeout=remaining)
            row.update(returncode=code, finished_utc=utc())
            require(code == 0, "Worker failed; no retry or subsequent candidate: " + name)
            if name == "train":
                require(load(out / "train" / "run" / "summary.json")["status"] == "epoch_complete", "Incomplete training")
            else:
                summary = load(out / "full-dev" / "summary.json")
                require(summary["status"] == "evaluation_complete" and summary["total"]["windows"] == 18792,
                        "Incomplete full dev")
                require(summary["total"]["loss_tokens"] == 17418742, "Incorrect dev targets")
                receipt = load(out / "full-dev-local-evaluation-audit.json")
                require(receipt["status"] == "complete_local_full_dev_audited"
                        and receipt["condition"] == "W_fixed_local", "Missing complete W evaluation audit")
            row["status"] = "complete"
            state["child_pid"] = None
            write(out / "stage.json", state)
        state.update(status="complete_pending_independent_local_audit", active_stage=None)
        return 0
    except BaseException as error:
        abnormal = True
        state.update(status="failed_or_incomplete", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state.update(finished_utc=utc(), elapsed_wall_seconds=time.monotonic() - started)
        state["worker_cost_estimate_usd"] = state["elapsed_wall_seconds"] / 3600 * 1.4
        state["cost_scope"] = "Execution subset, no overlapping Pod interval addition"
        try:
            write(out / "stage.json", state)
        finally:
            # A full evidence disk must not bypass failure cleanup.
            if abnormal:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                os.killpg(os.getpgrp(), signal.SIGTERM)
                time.sleep(2)
                os.killpg(os.getpgrp(), signal.SIGKILL)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    path = resolve(a.protocol)
    require(sha(path) == a.protocol_sha256, "Protocol hash mismatch")
    if a.execute:
        return execute(path, a.protocol_sha256)
    validate(load(path))
    print(json.dumps(dict(status="validated_no_model_calls", model_calls=0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
