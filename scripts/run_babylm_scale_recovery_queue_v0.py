"""Bounded recovery to the original final1413 point, then unchanged full dev.

This administrative queue contains no model operations, scientific changes,
retries, or checkpoint selection. Its two workers own the shared device lock.
A dedicated external timeout process group must bound this entire queue.
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
SELF = "scripts/run_babylm_scale_recovery_queue_v0.py"
STAGES = (
    ("train", "scripts/run_babylm_scale_epoch_recovery_v0.py", 5400),
    ("full_dev", "scripts/run_babylm_scale_full_dev_v0.py", 3600),
)
RECEIPTS = {
    "train": ("continuation-audit.json", "complete_epoch_continuation_audited"),
    "full_dev": ("full-dev-audit.json", "complete_full_dev_audited"),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def resolve(value):
    path = (ROOT / value).resolve()
    require(path.is_relative_to(ROOT), "Path escapes project")
    return path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(master):
    require(master["schema_version"] == 1 and master["launch_allowed"] is True, "Not frozen")
    require(master["hard_timeout_seconds"] == 10800, "Three-hour whole-group ceiling required")
    require(master["batch_cost_cap_usd"] == 4.2 and master["conservative_hourly_rate_usd"] == 1.4,
            "Frozen recovery execution cost bound changed")
    require(master["hard_timeout_seconds"] / 3600 * master["conservative_hourly_rate_usd"]
            <= master["batch_cost_cap_usd"], "Wall ceiling exceeds batch budget")
    pins = master["source_sha256"]
    require(pins.get(SELF) == sha(__file__), "Queue source unpinned")
    require(all(script in pins for _, script, _ in STAGES), "Both worker sources must be pinned")
    for name, digest in pins.items():
        require(sha(resolve(name)) == digest, "Source mismatch: " + name)
    require(set(master["stages"]) == {name for name, _, _ in STAGES}, "Unexpected extra experiment")
    out = resolve(master["output_dir"])
    require(out.is_relative_to(ROOT / "results") and out != ROOT / "results", "New bounded results tree required")
    protocols = {}
    for name, script, limit in STAGES:
        stage = master["stages"][name]
        require(stage["script"] == script and stage["timeout_seconds"] == limit, "Stage contract mismatch")
        protocol_path = resolve(stage["protocol"])
        require(protocol_path.is_relative_to(ROOT / "configs"), "Stage protocol must be under configs")
        require(sha(protocol_path) == stage["protocol_sha256"], "Stage protocol changed")
        protocol = load(protocol_path)
        require(protocol["hard_timeout_seconds"] == limit, "Worker hard bound differs from queue")
        stage_out = resolve(protocol["output_dir"])
        require(stage_out.is_relative_to(out) and stage_out != out, "Stage output not below queue evidence tree")
        protocols[name] = protocol
    train_out = resolve(protocols["train"]["output_dir"])
    eval_out = resolve(protocols["full_dev"]["output_dir"])
    require(not train_out.is_relative_to(eval_out) and not eval_out.is_relative_to(train_out), "Worker outputs overlap")
    evaluation = protocols["full_dev"]
    require(resolve(evaluation["continuation_master"]) == resolve(master["stages"]["train"]["protocol"])
            and evaluation["continuation_master_sha256"] == master["stages"]["train"]["protocol_sha256"],
            "Full dev is not bound to this recovery master")
    require(resolve(evaluation["training_run_dir"]) == train_out / "run"
            and resolve(evaluation["continuation_audit_path"]) == train_out / "continuation-audit.json",
            "Full dev points to another training branch")
    return out, protocols


def audit_stage(name, protocol, protocol_sha):
    filename, status = RECEIPTS[name]
    path = resolve(protocol["output_dir"]) / filename
    receipt = load(path)
    require(receipt.get("status") == status and receipt.get("master_protocol_sha256") == protocol_sha,
            "Successful exit lacks the matching completed " + name + " audit")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "status": status}


def execute(path, digest):
    require(os.name == "posix", "Cloud execution requires Linux")
    path = resolve(path)
    require(sha(path) == digest, "Master SHA mismatch")
    master = load(path)
    out, protocols = validate(master)
    require(os.environ.get("RUNPOD_POD_ID") == master["pod_id"], "Wrong Pod")
    require(os.getppid() == os.getpgrp(), "Controller must be direct child of its dedicated timeout group leader")
    require(not out.exists(), "Refusing overwrite or retry of queue evidence")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "master.json", master)
    started = time.monotonic()
    state = {"status": "starting", "pid": os.getpid(), "pgid": os.getpgrp(), "started_utc": utc(),
             "protocol_sha256": digest, "stages": [], "no_automatic_retry_or_followup": True,
             "queue_model_forward_calls": 0, "queue_backward_calls": 0, "queue_optimizer_updates": 0}
    child = None
    abnormal = False

    def interrupted(signum, frame):
        raise InterruptedError("Recovery queue received signal " + str(signum))

    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        for name, script, limit in STAGES:
            # Detect source/protocol edits between the two fixed stages.
            validate(master)
            entry = master["stages"][name]
            row = {"name": name, "started_utc": utc(), "status": "starting"}
            state["stages"].append(row)
            remaining = min(limit, master["hard_timeout_seconds"] - 60 - (time.monotonic() - started))
            require(remaining > 60, "No remaining bounded execution time")
            command = [sys.executable, "-u", str(resolve(script)), "--protocol", str(resolve(entry["protocol"])),
                       "--protocol-sha256", entry["protocol_sha256"], "--execute"]
            with (out / (name + ".stdout.log")).open("xb") as stdout, (out / (name + ".stderr.log")).open("xb") as stderr:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=False)
                row.update(status="running", pid=child.pid, pgid=os.getpgid(child.pid))
                require(row["pgid"] == os.getpgrp(), "Worker escaped the guarded process group")
                state.update(status="running", active_stage=name, child_pid=child.pid)
                write(out / "stage.json", state)
                code = child.wait(timeout=remaining)
            row.update(returncode=code, finished_utc=utc())
            require(code == 0, "Stage failed: " + name + "; no retry or subsequent stage")
            row["audit_receipt"] = audit_stage(name, protocols[name], entry["protocol_sha256"])
            row["status"] = "complete"
            child = None
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
        state["worker_cost_estimate_usd"] = state["elapsed_wall_seconds"] / 3600 * master["conservative_hourly_rate_usd"]
        state["cost_scope"] = "Queue execution subset only; excludes setup and idle; do not sum with overlapping Pod interval"
        if abnormal:
            # An already-dead wrapper can still leave a live model descendant.
            # Save evidence, then terminate the dedicated group including self.
            state["failure_cleanup"] = "dedicated_process_group_TERM_then_KILL"
            write(out / "stage.json", state)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            os.killpg(os.getpgrp(), signal.SIGTERM)
            time.sleep(2)
            os.killpg(os.getpgrp(), signal.SIGKILL)
        write(out / "stage.json", state)
        signal.signal(signal.SIGTERM, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(sha(resolve(args.protocol)) == args.protocol_sha256, "Master SHA mismatch")
    if args.execute:
        return execute(args.protocol, args.protocol_sha256)
    validate(load(resolve(args.protocol)))
    print(json.dumps({"status": "validated_no_model_calls", "model_forward_calls": 0,
                      "backward_calls": 0, "optimizer_updates": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
