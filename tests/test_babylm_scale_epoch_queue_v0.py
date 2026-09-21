"""No-model tests for finite queue sequencing and failed-stage containment."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import os
from scripts import run_babylm_scale_epoch_queue_v0 as q


class QueueTests(unittest.TestCase):
    def run_queue(self, codes):
        calls = []
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            (root / "configs").mkdir()
            stages = {}
            for name, script, limit in q.STAGES:
                path = root / "configs" / (name + ".json")
                path.write_text(json.dumps({"output_dir": "results/queue/" + name}))
                stages[name] = dict(script=script, timeout_seconds=limit,
                                    protocol="configs/" + name + ".json", protocol_sha256=q.sha(path))
            master = dict(schema_version=1, launch_allowed=True, hard_timeout_seconds=14400,
                          batch_cost_cap_usd=6, cycle_cost_cap_usd=20, conservative_hourly_rate_usd=1.4,
                          source_sha256={"scripts/run_babylm_scale_epoch_queue_v0.py": q.sha(q.__file__)},
                          stages=stages, output_dir="results/queue", pod_id="test-pod")
            (root / "scripts").mkdir()
            (root / "scripts/run_babylm_scale_epoch_queue_v0.py").write_bytes(Path(q.__file__).read_bytes())
            path = root / "configs/master.json"
            path.write_text(json.dumps(master))
            def popen(command, **kwargs):
                self.assertFalse(kwargs["start_new_session"])
                code = codes[len(calls)]
                calls.append(command)
                return SimpleNamespace(pid=100 + len(calls), wait=lambda timeout: code,
                                       poll=lambda: code)
            kills = []
            fake_os = SimpleNamespace(name="posix", environ={"RUNPOD_POD_ID": "test-pod"},
                                      getpid=lambda: 91, getppid=lambda: 90, getpgrp=lambda: 90, getpgid=lambda pid: 90,
                                      fsync=os.fsync, replace=os.replace, killpg=lambda p, s: kills.append((p, s)))
            with mock.patch.object(q, "ROOT", root), mock.patch.object(q, "os", fake_os), \
                 mock.patch.object(q.subprocess, "Popen", side_effect=popen), \
                 mock.patch.object(q.signal, "SIGKILL", 9, create=True), \
                 mock.patch.object(q.time, "sleep"):
                if any(codes):
                    with self.assertRaisesRegex(ValueError, "no retry or subsequent stage"):
                        q.execute("configs/master.json", q.sha(path))
                else:
                    self.assertEqual(q.execute("configs/master.json", q.sha(path)), 0)
                result = json.loads((root / "results/queue/stage.json").read_text())
            return calls, result, kills

    def test_failure_does_not_launch_evaluation(self):
        calls, state, kills = self.run_queue([2])
        self.assertEqual(len(calls), 1)
        self.assertEqual(state["status"], "failed_or_incomplete")
        self.assertEqual(len(kills), 2)

    def test_success_runs_only_fixed_two_stages(self):
        calls, state, kills = self.run_queue([0, 0])
        self.assertEqual(len(calls), 2)
        self.assertEqual(kills, [])
        self.assertEqual(state["status"], "complete_pending_independent_local_audit")
        self.assertEqual([s["name"] for s in state["stages"]], ["train", "full_dev"])

    def test_dead_wrapper_still_cleans_dedicated_group(self):
        calls, state, kills = self.run_queue([-9])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(kills), 2)
        self.assertEqual(state["failure_cleanup"], "dedicated_process_group_TERM_then_KILL")


if __name__ == "__main__":
    unittest.main()
