"""No-model metadata and mocked process-lifecycle tests for recovery queue."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import run_babylm_scale_recovery_queue_v0 as q


class RecoveryQueueTests(unittest.TestCase):
    def fixture(self, root):
        (root / "configs").mkdir()
        (root / "scripts").mkdir()
        (root / q.SELF).write_bytes(Path(q.__file__).read_bytes())
        pins = {q.SELF: q.sha(root / q.SELF)}
        protocols = {
            "train": {"output_dir": "results/recovery/train", "hard_timeout_seconds": 5400},
            "full_dev": {"output_dir": "results/recovery/full-dev", "hard_timeout_seconds": 3600,
                         "continuation_master": "configs/train.json", "training_run_dir": "results/recovery/train/run",
                         "continuation_audit_path": "results/recovery/train/continuation-audit.json"},
        }
        stages = {}
        for name, script, limit in q.STAGES:
            (root / script).write_text("# Never executed in unit tests.\n", encoding="utf-8")
            pins[script] = q.sha(root / script)
            if name == "full_dev":
                protocols[name]["continuation_master_sha256"] = stages["train"]["protocol_sha256"]
            path = root / "configs" / (name + ".json")
            path.write_text(json.dumps(protocols[name]), encoding="utf-8")
            stages[name] = {"script": script, "timeout_seconds": limit, "protocol": "configs/" + name + ".json",
                            "protocol_sha256": q.sha(path)}
        master = {"schema_version": 1, "launch_allowed": True, "hard_timeout_seconds": 10800,
                  "batch_cost_cap_usd": 4.2, "conservative_hourly_rate_usd": 1.4,
                  "source_sha256": pins, "stages": stages, "output_dir": "results/recovery", "pod_id": "test-pod"}
        path = root / "configs/master.json"
        path.write_text(json.dumps(master), encoding="utf-8")
        return master, path, protocols

    def exercise(self, codes, receipt_status=None, pod="test-pod", guarded=True):
        calls, kills, waits = [], [], []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            master, path, protocols = self.fixture(root)

            def popen(command, **kwargs):
                self.assertFalse(kwargs["start_new_session"])
                index = len(calls)
                name = q.STAGES[index][0]
                calls.append(command)
                receipt_dir = root / protocols[name]["output_dir"]
                receipt_dir.mkdir(parents=True)
                filename, status = q.RECEIPTS[name]
                receipt = {"status": receipt_status or status,
                           "master_protocol_sha256": master["stages"][name]["protocol_sha256"]}
                (receipt_dir / filename).write_text(json.dumps(receipt), encoding="utf-8")

                def wait(timeout):
                    waits.append(timeout)
                    if isinstance(codes[index], BaseException):
                        raise codes[index]
                    return codes[index]

                return SimpleNamespace(pid=100 + index, wait=wait)

            fake_os = SimpleNamespace(name="posix", environ={"RUNPOD_POD_ID": pod}, getpid=lambda: 91,
                getppid=lambda: 90 if guarded else 80, getpgrp=lambda: 90, getpgid=lambda pid: 90,
                fsync=os.fsync, replace=os.replace, killpg=lambda p, s: kills.append((p, s)))
            error = None
            with mock.patch.object(q, "ROOT", root), mock.patch.object(q, "os", fake_os), \
                 mock.patch.object(q.subprocess, "Popen", side_effect=popen), \
                 mock.patch.object(q.signal, "SIGKILL", 9, create=True), mock.patch.object(q.time, "sleep"):
                try:
                    result = q.execute("configs/master.json", q.sha(path))
                    self.assertEqual(result, 0)
                except BaseException as caught:
                    error = caught
                state_path = root / "results/recovery/stage.json"
                state = q.load(state_path) if state_path.exists() else None
            return calls, kills, waits, state, error

    def test_success_launches_only_two_fixed_same_group_stages(self):
        calls, kills, waits, state, error = self.exercise([0, 0])
        self.assertIsNone(error)
        self.assertEqual(len(calls), 2)
        self.assertEqual(kills, [])
        self.assertEqual(waits, [5400, 3600])
        self.assertEqual(state["status"], "complete_pending_independent_local_audit")
        self.assertEqual([row["name"] for row in state["stages"]], ["train", "full_dev"])
        self.assertTrue(all("audit_receipt" in row for row in state["stages"]))

    def test_failed_train_does_not_launch_evaluation(self):
        calls, kills, waits, state, error = self.exercise([2])
        self.assertIsInstance(error, ValueError)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(kills), 2)
        self.assertEqual(state["status"], "failed_or_incomplete")
        self.assertIn("worker_cost_estimate_usd", state)

    def test_dead_wrapper_still_terminates_whole_guarded_group(self):
        calls, kills, waits, state, error = self.exercise([-9])
        self.assertEqual(len(calls), 1)
        self.assertEqual(kills, [(90, q.signal.SIGTERM), (90, 9)])
        self.assertEqual(state["failure_cleanup"], "dedicated_process_group_TERM_then_KILL")

    def test_worker_timeout_prevents_eval_and_cleans_group(self):
        calls, kills, waits, state, error = self.exercise([q.subprocess.TimeoutExpired("mock worker", 5400)])
        self.assertIsInstance(error, q.subprocess.TimeoutExpired)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(kills), 2)

    def test_zero_exit_without_complete_audit_is_failure(self):
        calls, kills, waits, state, error = self.exercise([0], receipt_status="partial")
        self.assertIsInstance(error, ValueError)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(kills), 2)

    def test_wrong_pod_never_launches_or_signals(self):
        calls, kills, waits, state, error = self.exercise([0], pod="wrong")
        self.assertEqual(calls, [])
        self.assertEqual(kills, [])
        self.assertIsNone(state)
        self.assertIsInstance(error, ValueError)

    def test_absent_dedicated_guard_never_launches_or_signals(self):
        calls, kills, waits, state, error = self.exercise([0], guarded=False)
        self.assertEqual(calls, [])
        self.assertEqual(kills, [])
        self.assertIsNone(state)
        self.assertIsInstance(error, ValueError)

    def test_cost_and_stage_limits_cannot_expand(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            master, _, _ = self.fixture(root)
            with mock.patch.object(q, "ROOT", root):
                for key, value in (("hard_timeout_seconds", 14400), ("batch_cost_cap_usd", 6),
                                   ("conservative_hourly_rate_usd", 1.5)):
                    changed = copy.deepcopy(master)
                    changed[key] = value
                    with self.assertRaises(ValueError):
                        q.validate(changed)
                master["stages"]["train"]["timeout_seconds"] = 10800
                with self.assertRaisesRegex(ValueError, "Stage contract"):
                    q.validate(master)

    def test_changed_protocol_or_wrong_training_branch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            master, _, protocols = self.fixture(root)
            eval_path = root / master["stages"]["full_dev"]["protocol"]
            protocols["full_dev"]["training_run_dir"] = "results/another/run"
            eval_path.write_text(json.dumps(protocols["full_dev"]), encoding="utf-8")
            with mock.patch.object(q, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Stage protocol changed"):
                    q.validate(master)
                master["stages"]["full_dev"]["protocol_sha256"] = q.sha(eval_path)
                with self.assertRaisesRegex(ValueError, "another training branch"):
                    q.validate(master)


if __name__ == "__main__":
    unittest.main()
