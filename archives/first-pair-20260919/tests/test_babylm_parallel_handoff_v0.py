"""Mock-only handoff checks: no real children, signals, model, or GPU calls."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import check_babylm_parallel_handoff_v0 as handoff
from scripts import run_babylm_scientific_pair_v0 as launcher


class ParallelHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='babylm-handoff-mock-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for relative in handoff.AUDITED_SOURCE_HASHES:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / relative, destination)
        self.master = self.root / 'science-protocol.json'
        self.output = self.root / 'science'
        self.evidence = self.root / 'handoff-evidence'
        self.p = {'schema_version': 1, 'scope': 'scientific', 'device': 'cuda', 'dtype': 'float32',
                  'launch_allowed': True, 'hourly_rate_usd': .53, 'stage_spent_usd': .2,
                  'pair_stage_ceiling_usd': 150, 'paid_ceiling_usd': 150,
                  'max_wall_seconds': 100, 'hard_timeout_seconds_per_run': 110,
                  'expected_stop_reason': 'word_budget_reached'}
        launcher.write(self.master, self.p)
        self.original = self.master.read_bytes()
        self.sha = launcher.sha(self.master)
        self.initial = {'embedding.weight': 'a'*64}
        self.sources = {str(self.root / relative): expected for relative, expected in handoff.AUDITED_SOURCE_HASHES.items()}
        self.data = {'manifest_sha256': 'b'*64}
        self.identities = {
            100: {'pid': 100, 'ppid': 99, 'state': 'S', 'start_ticks': 123,
                  'cwd': str(self.root), 'argv': ['python', str(self.root / 'scripts/run_babylm_scientific_pair_v0.py'),
                  '--protocol', str(self.master), '--protocol-sha256', self.sha,
                  '--output-dir', str(self.output), '--execute']},
            101: {'pid': 101, 'ppid': 100, 'state': 'R', 'start_ticks': 124,
                  'cwd': str(self.root), 'argv': ['python', str(self.root / 'scripts/run_babylm_de_v0.py'),
                  '--protocol', str(self.output / 'dense-protocol.json'), '--mode', 'dense',
                  '--output-dir', str(self.output / 'dense')]}}
        self.calls = []

    def reader(self, pid):
        return json.loads(json.dumps(self.identities[pid]))

    def args(self):
        return (self.root, self.master, self.sha, self.output, 100, 101)

    def prepare_running(self):
        self.output.mkdir()
        (self.output / 'input-protocol.json').write_bytes(self.original)
        launcher.write(self.output / 'dense-protocol.json', {**self.p, 'stage_spent_usd': .201})
        launcher.write(self.output / 'stage.json', {'status': 'running_dense', 'protocol_sha256': self.sha,
            'jobs': [{'label': 'dense', 'child_protocol_file_sha256': launcher.sha(self.output / 'dense-protocol.json')}]})

    def test_01_plan_does_not_mutate_any_protocol(self):
        self.prepare_running()
        before = {str(x): x.read_bytes() for x in self.output.iterdir()}
        result = handoff.inspect_handoff(*self.args(), process_reader=self.reader)
        self.assertEqual(result['status'], 'handoff_plan_no_changes')
        self.assertEqual(self.master.read_bytes(), self.original)
        self.assertEqual(before, {str(x): x.read_bytes() for x in self.output.iterdir()})
        self.assertFalse(self.evidence.exists())

    def test_02_actual_parent_flow_completes_dense_then_refuses_sparse(self):
        def fake_child(command, **kwargs):
            mode = command[command.index('--mode') + 1]
            self.calls.append(mode)
            self.assertEqual(mode, 'dense')
            child = Path(command[command.index('--protocol') + 1])
            before = child.read_bytes()
            self.assertTrue(kwargs['timeout'] > 0)
            result = handoff.revoke_dispatch(*self.args(), self.evidence, process_reader=self.reader)
            self.assertTrue(result['revocation_confirmed'])
            self.assertFalse(self.master.exists())
            self.assertEqual(child.read_bytes(), before)
            self.assertEqual((self.output / 'input-protocol.json').read_bytes(), self.original)
            self.assertEqual((self.evidence / 'revoked-master-protocol.json').read_bytes(), self.original)
            # D continues reading its own protocol and publishes a valid result
            # after the dispatch master has been revoked. No fake training loop.
            p = launcher.load(child)
            destination = self.output / 'dense'; destination.mkdir()
            launcher.write(destination / 'summary.json', {'scope': 'scientific', 'mode': 'dense',
                'status': 'word_budget_reached', 'protocol_sha256': launcher.protocol_hash(p),
                'source_hashes': self.sources, 'data_fingerprint': self.data,
                'initial_parameter_hashes': self.initial,
                'counts': {'engineering_updates': 0, 'updates': 0}, 'cursor': {}})
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(launcher, 'ROOT', self.root), \
             mock.patch.object(launcher, 'verify_preflight', return_value=({}, self.sources, self.initial, self.data)), \
             mock.patch.object(launcher.subprocess, 'run', side_effect=fake_child), \
             mock.patch.object(handoff.os, 'kill', side_effect=AssertionError('No signals permitted')):
            with self.assertRaises(FileNotFoundError):
                launcher.launch(self.master, self.sha, self.output, execute=True)
        self.assertEqual(self.calls, ['dense'])
        stage = launcher.load(self.output / 'stage.json')
        self.assertEqual(stage['status'], 'stopped_with_failure')
        self.assertEqual(stage['jobs'][0]['status'], 'completed')
        self.assertEqual(stage['jobs'][0]['returncode'], 0)
        self.assertEqual(stage['error_type'], 'FileNotFoundError')
        stage_bytes = (self.output / 'stage.json').read_bytes()
        terminal = handoff.verify_terminal(self.evidence)
        self.assertEqual(terminal['sparse_jobs_launched'], 0)
        self.assertFalse(terminal['scientific_pair_complete'])
        self.assertEqual((self.output / 'stage.json').read_bytes(), stage_bytes)
        for name in ('babylm-scientific-pair.lock', 'babylm-gpu-preflight.lock'):
            self.assertFalse((self.root / 'logs' / name).exists())

    def test_03_stale_pid_binding_or_dispatch_state_refused_before_move(self):
        self.prepare_running()
        self.identities[101]['ppid'] = 999
        with self.assertRaisesRegex(ValueError, 'direct child'):
            handoff.revoke_dispatch(*self.args(), self.evidence, process_reader=self.reader)
        self.assertEqual(self.master.read_bytes(), self.original)
        self.identities[101]['ppid'] = 100
        stage = launcher.load(self.output / 'stage.json')
        stage['jobs'][0]['returncode'] = 0
        launcher.write(self.output / 'stage.json', stage)
        with self.assertRaisesRegex(ValueError, 'waiting'):
            handoff.revoke_dispatch(*self.args(), self.evidence, process_reader=self.reader)
        self.assertTrue(self.master.exists()); self.assertFalse(self.evidence.exists())

    def test_04_dense_exits_during_rename_is_ambiguous_not_success(self):
        self.prepare_running()
        def raced_reader(pid):
            record = self.reader(pid)
            if pid == 101 and not self.master.exists():
                record['state'] = 'Z'
            return record
        with self.assertRaisesRegex(RuntimeError, 'non-zombie'):
            handoff.revoke_dispatch(*self.args(), self.evidence, process_reader=raced_reader)
        failure = handoff.load(self.evidence / 'failure.json')
        self.assertTrue(failure['moved']); self.assertFalse(failure['revocation_confirmed'])
        self.assertIn('do_not_launch_second_sparse', failure['status'])
        self.assertIn('Traceback', failure['traceback'])
        self.assertFalse(self.master.exists())
        self.assertEqual((self.evidence / 'revoked-master-protocol.json').read_bytes(), self.original)
        self.assertFalse((self.evidence / 'revocation.json').exists())

    def test_05_source_or_child_sha_change_refused(self):
        self.prepare_running()
        child = self.output / 'dense-protocol.json'
        child.write_bytes(child.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'Dense child protocol SHA'):
            handoff.inspect_handoff(*self.args(), process_reader=self.reader)
        self.assertTrue(self.master.exists())
        source = self.root / 'scripts/run_babylm_de_v0.py'
        source.write_bytes(source.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'Unaudited'):
            handoff.inspect_handoff(*self.args(), process_reader=self.reader)

    def test_06_existing_evidence_is_not_overwritten(self):
        self.prepare_running()
        self.evidence.mkdir(); sentinel = self.evidence / 'original'; sentinel.write_bytes(b'evidence')
        with self.assertRaises(FileExistsError):
            handoff.revoke_dispatch(*self.args(), self.evidence, process_reader=self.reader)
        self.assertEqual(sentinel.read_bytes(), b'evidence')
        self.assertEqual(self.master.read_bytes(), self.original)


if __name__ == '__main__':
    stream = io.StringIO()
    selected = sys.argv[1:]
    suite = (unittest.TestSuite(ParallelHandoffTests(name) for name in selected) if selected else
             unittest.defaultTestLoader.loadTestsFromTestCase(ParallelHandoffTests))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report_path = PROJECT / 'results/babylm-parallel-handoff-mock-v0.json'
    prior = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else None
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'passed': result.wasSuccessful(),
              'tests_run': result.testsRun, 'selected_tests': selected or 'all', 'raw_test_output': stream.getvalue(),
              'failures': [str(x) for x in result.failures], 'errors': [str(x) for x in result.errors],
              'actual_subprocesses': 0, 'signals_sent': 0, 'model_forwards': 0,
              'backwards': 0, 'optimizer_updates': 0, 'gpu_calls': 0, 'network_calls': 0,
              'source_sha256': handoff.sha(PROJECT / 'scripts/check_babylm_parallel_handoff_v0.py'),
              'launcher_sha256': launcher.sha(PROJECT / 'scripts/run_babylm_scientific_pair_v0.py')}
    if prior is not None:
        report['prior_attempt'] = prior
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
