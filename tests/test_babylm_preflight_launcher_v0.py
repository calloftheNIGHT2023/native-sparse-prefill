"""Launcher control-flow tests; all GPU/model commands are intercepted.

One timeout fixture runs only a short Python sleep child on CPU. No torch/model,
GPU, network, remote shell, rentals, or real training entrypoints are executed.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from scripts import run_babylm_gpu_preflight_pair_v0 as launcher

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT / 'logs/babylm-preflight-launcher-tests-current'
MEASUREMENTS = {'real_cpu_sleep_subprocesses': 0, 'model_calls': 0, 'gpu_calls': 0, 'network_calls': 0}
REAL_SUBPROCESS_RUN = subprocess.run


class PreflightLauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = ARTIFACT_ROOT / self._testMethodName
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fake_project = self.directory / 'project'
        (self.fake_project / 'logs').mkdir(parents=True, exist_ok=True)
        self.output = self.directory / 'pair'
        self.protocol_path = self.directory / 'protocol.json'
        self.calls = []
        self.stub_behavior = None

    def protocol(self):
        p = json.loads((PROJECT / 'configs/babylm-de-gpu-preflight-v0.draft.json').read_text(encoding='utf-8'))
        p.update(launch_allowed=True, budget_reconciled=True, hourly_rate_usd=0.74,
                 stage_spent_usd=0.25, pair_stage_ceiling_usd=10.0,
                 paid_ceiling_usd=10.0, remaining_research_budget_usd=30.0)
        return p

    def child_stub(self, command, **kwargs):
        script = Path(command[1]).name
        if script == 'check_babylm_gpu_numerics_v0.py':
            label = 'gpu-numerics'
        else:
            self.assertEqual(script, 'run_babylm_de_v0.py')
            label = command[command.index('--mode') + 1]
        self.calls.append(label)
        self.assertGreater(kwargs['timeout'], 0)
        self.assertFalse(kwargs['check'])
        self.assertEqual(Path(kwargs['cwd']), self.fake_project)
        behavior = self.stub_behavior(label, command, kwargs) if self.stub_behavior else None
        if behavior is not None:
            return behavior
        if label == 'gpu-numerics':
            Path(command[command.index('--output') + 1]).write_text('{"passed":true}', encoding='utf-8')
        else:
            destination = Path(command[command.index('--output-dir') + 1])
            destination.mkdir()
            (destination / 'summary.json').write_text('{"status":"max_updates_reached"}', encoding='utf-8')
        return subprocess.CompletedProcess(command, 0)

    def invoke(self, protocol=None, execute=True, write_override=None):
        self.protocol_path.write_text(json.dumps(protocol or self.protocol()), encoding='utf-8')
        argv = ['launcher', '--protocol', str(self.protocol_path), '--output-dir', str(self.output)]
        if execute:
            argv.append('--execute')
        stream = io.StringIO()
        with mock.patch.object(launcher, 'ROOT', self.fake_project), mock.patch.object(sys, 'argv', argv), \
             mock.patch.object(launcher.subprocess, 'run', side_effect=self.child_stub), contextlib.redirect_stdout(stream):
            if write_override is not None:
                with mock.patch.object(launcher, 'write', side_effect=write_override):
                    result = launcher.main()
            else:
                result = launcher.main()
        return result, stream.getvalue()

    def stage(self):
        return json.loads((self.output / 'stage.json').read_text(encoding='utf-8'))

    def test_01_default_dry_run_starts_nothing(self):
        p = self.protocol(); p.update(launch_allowed=False, budget_reconciled=False,
                                     hourly_rate_usd=None, stage_spent_usd=None)
        result, stdout = self.invoke(p, execute=False)
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout)['status'], 'plan_only_no_processes_started')
        self.assertFalse(self.calls)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.fake_project / 'logs/babylm-gpu-preflight.lock').exists())

    def test_02_unauthorized_null_and_insufficient_budget_refused(self):
        variants = [dict(launch_allowed=False), dict(budget_reconciled=False),
                    dict(hourly_rate_usd=None), dict(stage_spent_usd=None),
                    dict(remaining_research_budget_usd=None), dict(remaining_research_budget_usd=0.1),
                    dict(pair_stage_ceiling_usd=10.01, paid_ceiling_usd=10.01),
                    dict(paid_ceiling_usd=9.0), dict(hourly_rate_usd=float('nan')),
                    dict(eval_initial=True)]
        for changes in variants:
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                self.invoke({**self.protocol(), **changes})
            self.assertFalse(self.calls)

    def test_03_numerics_dense_sparse_order_and_rolling_spend(self):
        sentinel = self.fake_project / 'logs/unrelated-job.lock'
        sentinel.write_text('owned by another job', encoding='utf-8')
        result, _ = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(self.calls, ['gpu-numerics', 'dense', 'sparse'])
        record = self.stage()
        self.assertEqual(record['status'], 'paired_preflight_complete_not_scientific_training')
        self.assertEqual([job['label'] for job in record['jobs']], self.calls)
        self.assertTrue(all(job['returncode'] == 0 for job in record['jobs']))
        dense = json.loads((self.output / 'dense-protocol.json').read_text())
        sparse = json.loads((self.output / 'sparse-protocol.json').read_text())
        self.assertGreaterEqual(dense['stage_spent_usd'], 0.25)
        self.assertGreaterEqual(sparse['stage_spent_usd'], dense['stage_spent_usd'])
        self.assertEqual(sentinel.read_text(), 'owned by another job')
        self.assertFalse((self.fake_project / 'logs/babylm-gpu-preflight.lock').exists())

    def test_04_gpu_numerics_requires_literal_true(self):
        for passed in (False, 'true', 1):
            self.output = self.directory / ('pair-' + str(passed))
            self.calls = []
            def behavior(label, command, kwargs):
                if label == 'gpu-numerics':
                    Path(command[command.index('--output') + 1]).write_text(json.dumps({'passed': passed}))
                    return subprocess.CompletedProcess(command, 0)
            self.stub_behavior = behavior
            with self.subTest(passed=passed), self.assertRaisesRegex(RuntimeError, 'explicitly pass'):
                self.invoke()
            self.assertEqual(self.calls, ['gpu-numerics'])
            self.assertEqual(self.stage()['status'], 'stopped_with_failure')
            self.assertFalse((self.fake_project / 'logs/babylm-gpu-preflight.lock').exists())

    def test_05_dense_process_failure_does_not_start_sparse(self):
        self.stub_behavior = lambda label, command, kwargs: subprocess.CompletedProcess(command, 17) if label == 'dense' else None
        with self.assertRaisesRegex(RuntimeError, 'dense failed'):
            self.invoke()
        self.assertEqual(self.calls, ['gpu-numerics', 'dense'])
        self.assertEqual(self.stage()['jobs'][-1]['returncode'], 17)
        self.assertEqual(self.stage()['status'], 'stopped_with_failure')
        self.assertFalse((self.fake_project / 'logs/babylm-gpu-preflight.lock').exists())

    def test_06_incomplete_dense_summary_does_not_start_sparse(self):
        def behavior(label, command, kwargs):
            if label == 'dense':
                destination = Path(command[command.index('--output-dir') + 1]); destination.mkdir()
                (destination / 'summary.json').write_text('{"status":"word_budget_reached"}')
                return subprocess.CompletedProcess(command, 0)
        self.stub_behavior = behavior
        with self.assertRaisesRegex(RuntimeError, 'did not complete'):
            self.invoke()
        self.assertEqual(self.calls, ['gpu-numerics', 'dense'])
        self.assertEqual(self.stage()['status'], 'stopped_with_failure')

    def test_07_cpu_sleep_timeout_is_recorded_and_unlocks(self):
        def behavior(label, command, kwargs):
            self.assertEqual(label, 'gpu-numerics')
            MEASUREMENTS['real_cpu_sleep_subprocesses'] += 1
            return REAL_SUBPROCESS_RUN([sys.executable, '-c', 'import time; print("CPU sleep fixture",flush=True); time.sleep(5)'],
                                       stdout=kwargs['stdout'], stderr=kwargs['stderr'], timeout=0.1, check=False)
        self.stub_behavior = behavior
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke()
        self.assertEqual(self.calls, ['gpu-numerics'])
        record = self.stage()
        self.assertEqual(record['error_type'], 'TimeoutExpired')
        self.assertEqual(record['jobs'][0]['status'], 'hard_timeout_child_killed_and_waited')
        self.assertFalse((self.fake_project / 'logs/babylm-gpu-preflight.lock').exists())

    def test_08_existing_global_lock_is_not_removed(self):
        lock = self.fake_project / 'logs/babylm-gpu-preflight.lock'
        lock.write_text('pre-existing owner lock', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(lock.read_text(), 'pre-existing owner lock')
        self.assertFalse(self.calls)

    def test_09_existing_output_evidence_is_not_overwritten(self):
        self.output.mkdir()
        evidence = self.output / 'evidence.txt'
        evidence.write_text('preserve prior result', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'fresh output'):
            self.invoke()
        self.assertEqual(evidence.read_text(), 'preserve prior result')
        self.assertFalse(self.calls)

    def test_10_exhausted_time_equivalent_budget_starts_no_child(self):
        p = self.protocol(); p['stage_spent_usd'] = 9.99999
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
            self.invoke(p)
        self.assertFalse(self.calls)
        self.assertEqual(self.stage()['status'], 'stopped_with_failure')
        self.assertFalse((self.fake_project / 'logs/babylm-gpu-preflight.lock').exists())

    def test_11_final_record_write_failure_still_releases_own_lock(self):
        original_write = launcher.write
        def failing_final_write(path, value):
            if path.name == 'stage.json' and value.get('status') == 'paired_preflight_complete_not_scientific_training':
                raise OSError('synthetic final stage write failure')
            return original_write(path, value)
        with self.assertRaisesRegex(OSError, 'synthetic final stage write failure'):
            self.invoke(write_override=failing_final_write)
        self.assertFalse((self.fake_project / 'logs/babylm-gpu-preflight.lock').exists(),
                         'A final audit-write exception must not leave this launcher lock behind')


if __name__ == '__main__':
    unittest.main(verbosity=2)
