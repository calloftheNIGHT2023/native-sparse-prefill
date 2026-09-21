"""Finite authorized scale-only continuation and fixed final full-dev evaluation.

No model code, retries, hyperparameter choices, or checkpoint selection here.
An external process-group timeout must cover this controller and its children.
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
STAGES = (
    ("train", "scripts/run_babylm_scale_epoch_continuation_v0.py", 10800),
    ("full_dev", "scripts/run_babylm_scale_full_dev_v0.py", 3600),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise ValueError(message)


def resolve(value):
    path = (ROOT / value).resolve()
    require(path.is_relative_to(ROOT), "Path escapes project")
    return path


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def validate(master):
    require(master["schema_version"] == 1 and master["launch_allowed"] is True, "Not frozen")
    require(master["hard_timeout_seconds"] == 14400, "Four-hour whole-group ceiling required")
    require(master["batch_cost_cap_usd"] == 6 and master["cycle_cost_cap_usd"] == 20,
            "Cost limits changed")
    require(master["conservative_hourly_rate_usd"] == 1.4, "Frozen rate changed")
    require(master["hard_timeout_seconds"] / 3600 * master["conservative_hourly_rate_usd"]
            <= master["batch_cost_cap_usd"], "Wall ceiling exceeds batch budget")
    require(master["source_sha256"].get("scripts/run_babylm_scale_epoch_queue_v0.py") == sha(__file__),
            "Queue source unpinned")
    for name, digest in master["source_sha256"].items():
        require(sha(resolve(name)) == digest, "Source mismatch: " + name)
    require(set(master["stages"]) == {name for name, _, _ in STAGES}, "Unexpected extra experiment")
    out = resolve(master["output_dir"])
    for name, script, limit in STAGES:
        stage = master["stages"][name]
        require(stage["script"] == script and stage["timeout_seconds"] == limit, "Stage contract mismatch")
        require(sha(resolve(stage["protocol"])) == stage["protocol_sha256"], "Stage protocol changed")
        protocol = json.loads(resolve(stage["protocol"]).read_text(encoding="utf-8"))
        require(resolve(protocol["output_dir"]).is_relative_to(out), "Stage output not in queue evidence tree")
    return out


def execute(path, digest):
    require(os.name == "posix", "Cloud execution requires Linux")
    path = resolve(path)
    require(sha(path) == digest, "Master SHA mismatch")
    master = json.loads(path.read_text(encoding="utf-8"))
    out = validate(master)
    require(os.environ.get("RUNPOD_POD_ID") == master["pod_id"], "Wrong Pod")
    require(os.getppid() == os.getpgrp(), "Controller must be direct child of its dedicated timeout group leader")
    require(not out.exists(), "Refusing overwrite/retry of queue evidence")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "master.json", master)
    started = time.monotonic()
    state = {"status": "starting", "pid": os.getpid(), "pgid": os.getpgrp(),
             "started_utc": utc(), "protocol_sha256": digest, "stages": [],
             "no_automatic_retry_or_followup": True}
    child = None
    abnormal = False
    def interrupted(signum, frame):
        raise InterruptedError("Queue received signal " + str(signum))
    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        for name, script, limit in STAGES:
            entry = master["stages"][name]
            row = {"name": name, "started_utc": utc(), "status": "starting"}
            state["stages"].append(row)
            remaining = min(limit, master["hard_timeout_seconds"] - 60 - (time.monotonic() - started))
            require(remaining > 60, "No remaining bounded execution time")
            command = [sys.executable, "-u", str(resolve(script)), "--protocol", str(resolve(entry["protocol"])),
                       "--protocol-sha256", entry["protocol_sha256"], "--execute"]
            with (out / (name + ".stdout.log")).open("xb") as so, (out / (name + ".stderr.log")).open("xb") as se:
                child = subprocess.Popen(command, cwd=ROOT, stdout=so, stderr=se, start_new_session=False)
                row.update(status="running", pid=child.pid, pgid=os.getpgid(child.pid))
                state.update(status="running", active_stage=name, child_pid=child.pid)
                write(out / "stage.json", state)
                code = child.wait(timeout=remaining)
            row.update(returncode=code, finished_utc=utc())
            require(code == 0, "Stage failed: " + name + "; no retry or subsequent stage")
            row["status"] = "complete"
            child = None
            write(out / "stage.json", state)
        state.update(status="complete_pending_independent_local_audit", active_stage=None)
        return 0
    except BaseException as exc:
        abnormal = True
        state.update(status="failed_or_incomplete", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if abnormal:
            # A dead stage wrapper can leave a live model child. Save failure
            # evidence first, then end this launcher's dedicated whole group.
            state.update(finished_utc=utc(), elapsed_wall_seconds=time.monotonic() - started,
                         failure_cleanup="dedicated_process_group_TERM_then_KILL")
            write(out / "stage.json", state)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            os.killpg(os.getpgrp(), signal.SIGTERM)
            time.sleep(2)
            os.killpg(os.getpgrp(), signal.SIGKILL)
        elif child is not None and child.poll() is None:
            # Stage wrappers catch TERM and stop their one same-group model child.
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                # Fail closed: stop the entire group, including any unresponsive
                # descendant, rather than leaving a model running without a guard.
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                os.killpg(os.getpgrp(), signal.SIGTERM)
                time.sleep(2)
                os.killpg(os.getpgrp(), signal.SIGKILL)
        state.update(finished_utc=utc(), elapsed_wall_seconds=time.monotonic() - started)
        state["worker_cost_estimate_usd"] = state["elapsed_wall_seconds"] / 3600 * master["conservative_hourly_rate_usd"]
        state["cost_scope"] = "Execution subset, excludes prior setup and idle; do not add to overlapping Pod interval"
        write(out / "stage.json", state)
        signal.signal(signal.SIGTERM, old_handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--protocol-sha256")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    if a.execute:
        require(bool(a.protocol_sha256), "Explicit protocol SHA required")
        return execute(a.protocol, a.protocol_sha256)
    validate(json.loads(resolve(a.protocol).read_text(encoding="utf-8")))
    print(json.dumps({"status": "validated", "model_forward_calls": 0, "backward_calls": 0, "optimizer_updates": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
