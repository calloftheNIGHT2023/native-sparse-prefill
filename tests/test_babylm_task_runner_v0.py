"""Fake-only runner gates/accounting. Never call HybridLM.forward or build it."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts import run_babylm_task_evaluation_v0 as runner
from babylm_hybrid.official_causal_scoring import CausalScoringError

STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
ARTIFACT_ROOT = ROOT / 'logs' / ('babylm-task-runner-fakes-' + STAMP)
COUNTS = {'mock_score_calls': 0, 'mock_model_constructions': 0,
          'actual_model_constructions': 0, 'actual_model_forward_calls': 0,
          'backwards': 0, 'optimizer_updates': 0, 'gpu_calls': 0, 'network_calls': 0}


class FakeStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3))

    def forward(self, *args, **kwargs):
        raise AssertionError('No model forward is permitted in these tests')


def fake_build(*args, **kwargs):
    COUNTS['mock_model_constructions'] += 1
    return FakeStateModel()


class TaskRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = ARTIFACT_ROOT / self._testMethodName
        self.directory.mkdir(parents=True)
        self.raws = [{'synthetic': True, 'sentence_good': f'The bird number {i} sings.',
                      'sentence_bad': f'The bird number {i} sing.', 'field': 'syntax',
                      'UID': 'toy_subject_agreement', 'linguistics_term': 'agreement'} for i in range(4)]
        self.raw_path = self.directory / 'toy.jsonl'
        self.raw_path.write_bytes(b''.join(runner.canonical(row) + b'\r\n' for row in self.raws))
        self.manifest_path = self.directory / 'manifest.json'
        self.manifest = {'schema_version': 1, 'scope': 'engineering_fixture', 'upstream_commit': runner.UPSTREAM_COMMIT,
                         'files': [{'path': 'toy.jsonl', 'sha256': runner.sha(self.raw_path), 'task': 'blimp',
                                    'split': 'synthetic', 'records': 4, 'synthetic': True}]}
        self.manifest_path.write_bytes(runner.canonical(self.manifest))
        self.p = {'schema_version': 1, 'scope': 'engineering_fixture', 'device': 'cpu',
                  'snapshot': str(self.directory / 'mock-not-loaded.pt'), 'snapshot_sha256': '1'*64,
                  'snapshot_metadata_sha256': '2'*64, 'training_protocol_sha256': '3'*64,
                  'mode': 'sparse', 'tokenizer': 'data/babylm-2026-tokenizer-16k-v0/tokenizer.json',
                  'tokenizer_sha256': runner.TOKENIZER_SHA256, 'task_manifest': str(self.manifest_path),
                  'task_manifest_sha256': runner.sha(self.manifest_path), 'seed': 0, 'temperature': 1.0,
                  'batch_size': 2, 'max_items': 4, 'max_forward_attempts': 4,
                  'max_wall_seconds': 30, 'max_input_tokens': 128, 'max_model_parameters': 1_000_000}

    def protocol(self, p=None, name='protocol.json'):
        path = self.directory / name
        path.write_bytes(runner.canonical(self.p if p is None else p))
        return path, runner.sha(path)

    def fake_score(self, model, processor, records, task, **kwargs):
        COUNTS['mock_score_calls'] += 1
        n = len(records)
        counters = {key: 0 for key in runner.COUNT_KEYS}
        counters.update(model_forward_attempts=2, model_forward_calls=2, records=n, candidate_sequences=2*n,
                        source_tokens=10*n, forward_input_tokens=8*n, padded_forward_positions=8*n,
                        scored_target_tokens=4*n, submitted_forward_input_tokens=8*n,
                        submitted_padded_forward_positions=8*n)
        return {'upstream_commit': runner.UPSTREAM_COMMIT, 'task': task, 'temperatures': [1.0],
                'normalization': 'completion_token_sum',
                'scores': [[[-1.0, -2.0] for _ in records]], 'scored_target_counts': [[2, 2] for _ in records],
                'scorable_candidates': [[True, True] for _ in records],
                'ranking_allowed_per_record': [True]*n, 'counters': counters,
                'tokenizer_sha256': runner.TOKENIZER_SHA256, 'official_benchmark_executed': False}

    def mocked_run(self, p=None, output='out', scorer=None):
        path, digest = self.protocol(p)
        with mock.patch.object(runner, 'load_snapshot_model', return_value=(object(), {'fixture': True})), \
             mock.patch('babylm_hybrid.official_causal_scoring.FixedLocalTokenizer', return_value=object()), \
             mock.patch('babylm_hybrid.official_causal_scoring.score_records', side_effect=scorer or self.fake_score):
            return runner.run_evaluation(path, digest, self.directory / output, execute=True)

    def test_01_default_plan_and_no_paid_or_scientific_execute(self):
        path, digest = self.protocol()
        with mock.patch.object(runner, 'load_records') as reader, mock.patch.object(runner, 'load_snapshot_model') as loader:
            plan = runner.run_evaluation(path, digest, self.directory/'unused')
            self.assertEqual(plan['status'], 'plan_only_no_model_or_task_data_loaded')
            reader.assert_not_called(); loader.assert_not_called()
            self.assertFalse((self.directory/'unused').exists())
            for index, changed in enumerate(({'device': 'cuda'}, {'scope': 'scientific'}, {'max_items': 129},
                                             {'max_forward_attempts': 65}, {'max_model_parameters': 1_000_001})):
                path, digest = self.protocol({**self.p, **changed}, f'gate{index}.json')
                with self.assertRaises(ValueError):
                    runner.run_evaluation(path, digest, self.directory/f'blocked{index}', execute=True)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                runner.run_evaluation(path, '0'*64, self.directory/'bad-hash', execute=True)
            reader.assert_not_called(); loader.assert_not_called()

    def test_02_mock_scores_real_reader_real_aggregator_and_crlf(self):
        result = self.mocked_run()
        out = self.directory/'out'
        self.assertEqual(result['counts']['model_forward_attempts'], 4)
        self.assertEqual(result['counts']['model_forward_calls'], 4)
        self.assertEqual(result['records_completed'], 4)
        records = [json.loads(row) for row in (out/'records.jsonl').read_text(encoding='utf-8').splitlines()]
        expected = hashlib.sha256(runner.canonical(self.raws[0]) + b'\r\n').hexdigest()
        self.assertEqual(records[0]['source']['raw_line_sha256'], expected)
        raw = [json.loads(row) for row in (out/'raw-scores.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(raw), 2)
        aggregated = json.loads((out/'predictions-and-aggregation.json').read_text(encoding='utf-8'))
        self.assertEqual(aggregated['results'][0]['official_uid_mean_accuracy_pct'], 100.0)
        self.assertEqual(len(aggregated['results'][0]['items']), 4)
        self.assertFalse((out/'run.lock').exists())
        with self.assertRaisesRegex(ValueError, 'already contains'):
            self.mocked_run()
        self.assertEqual(len((out/'raw-scores.jsonl').read_text(encoding='utf-8').splitlines()), 2)

    def test_03_second_batch_failure_retains_raw_partial_predictions_and_counts(self):
        calls = []
        def failing_score(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return self.fake_score(*args, **kwargs)
            COUNTS['mock_score_calls'] += 1
            counters = {key: 0 for key in runner.COUNT_KEYS}
            counters.update(model_forward_attempts=2, model_forward_calls=1, records=2, candidate_sequences=4,
                            source_tokens=20, submitted_forward_input_tokens=16, submitted_padded_forward_positions=16,
                            forward_input_tokens=8, padded_forward_positions=8, scored_target_tokens=4)
            raise CausalScoringError('injected candidate failure', counters=counters, task='blimp', candidate_index=1)
        with self.assertRaisesRegex(CausalScoringError, 'injected candidate'):
            self.mocked_run(scorer=failing_score)
        out = self.directory/'out'
        failure = json.loads((out/'failure.json').read_text(encoding='utf-8'))
        self.assertEqual(failure['counts']['model_forward_attempts'], 4)
        self.assertEqual(failure['counts']['model_forward_calls'], 3)
        self.assertEqual(failure['records_completed'], 2)
        self.assertEqual(failure['active_batch']['failed_candidate_index'], 1)
        self.assertIn('Traceback', failure['traceback'])
        self.assertEqual(len((out/'raw-scores.jsonl').read_text(encoding='utf-8').splitlines()), 1)
        self.assertTrue((out/'partial-predictions-and-aggregation.json').is_file())
        self.assertFalse((out/'run.lock').exists())

    def test_04_caps_tamper_synthetic_flags_and_existing_lock(self):
        for index, changes in enumerate(({'max_items': 3}, {'max_forward_attempts': 3},
                                          {'task_manifest_sha256': '0'*64})):
            with self.assertRaises(ValueError):
                self.mocked_run({**self.p, **changes}, output=f'limit{index}')
            self.assertTrue((self.directory/f'limit{index}'/'failure.json').exists())
        self.raws[0]['synthetic'] = False
        self.raw_path.write_bytes(b''.join(runner.canonical(row)+b'\n' for row in self.raws))
        self.manifest['files'][0]['sha256'] = runner.sha(self.raw_path)
        self.manifest_path.write_bytes(runner.canonical(self.manifest))
        with self.assertRaisesRegex(ValueError, 'synthetic=true'):
            self.mocked_run({**self.p, 'task_manifest_sha256': runner.sha(self.manifest_path)}, output='nonsynthetic')
        locked = self.directory/'locked'
        locked.mkdir()
        (locked/'run.lock').write_text('another owner', encoding='utf-8')
        path, digest = self.protocol()
        with self.assertRaises(FileExistsError):
            runner.run_evaluation(path, digest, locked, execute=True)
        self.assertEqual((locked/'run.lock').read_text(encoding='utf-8'), 'another owner')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            runner.strict_json('{"synthetic":true,"synthetic":false}')

    def test_05_safe_snapshot_loader_strict_provenance_and_preconstruction_gate(self):
        from babylm_hybrid.config import HybridConfig
        from babylm_hybrid import model as core_model
        from babylm_hybrid.training import source_hashes
        train = {'schema_version': 1, 'scope': 'engineering_smoke', 'launch_allowed': True, 'device': 'cpu',
                 'dtype': 'float32', 'model_config': HybridConfig(vocab_size=16384).to_dict(),
                 'backbone_seed': 1, 'indexer_seed': 2, 'data_order_seed': 3, 'train_manifest': 'fixture',
                 'train_manifest_sha256': 'a'*64, 'max_word_exposures': 2, 'max_updates': 1,
                 'max_wall_seconds': 10, 'windows_per_update': 1, 'learning_rate': 1e-3,
                 'indexer_learning_rate': 1e-3, 'weight_decay': 0, 'betas': [.9,.95], 'eps': 1e-8,
                 'grad_clip_norm': 1, 'aux_weight': 1, 'warmup_word_exposures': 0, 'min_lr_ratio': .1,
                 'checkpoint_every_updates': 1, 'eval_every_updates': 0}
        fake = FakeStateModel()
        param = fake.weight
        initial = {'weight': hashlib.sha256(runner.canonical({'shape':[3], 'dtype':'torch.float32'}) + param.detach().numpy().tobytes()).hexdigest()}
        counts = {'updates':0, 'word_exposures':0, 'input_tokens':0, 'loss_tokens':0}
        saved = {'snapshot_schema_version':1, 'kind':'model_only', 'resume_supported':False, 'point':counts.copy(),
                 'counts':counts, 'cursor':{'epoch':0,'position':0}, 'mode':'sparse', 'nominal_crossed_milestones':[],
                 'protocol_sha256':hashlib.sha256(runner.canonical(train)).hexdigest(), 'source_hashes':source_hashes(),
                 'data_fingerprint':{'tokenizer_sha256':runner.TOKENIZER_SHA256, 'manifest_sha256':'a'*64},
                 'initial_parameter_hashes':initial, 'saved_utc':runner.utc(), 'word_accounting':'fixture',
                 'purpose':'fake-state-only loader test', 'protocol':train, 'model_state':fake.state_dict()}
        def publish(value, stem):
            path = self.directory/(stem+'.pt')
            torch.save(value,path)
            metadata = {k:v for k,v in value.items() if k not in ('model_state','protocol')}
            metadata.update(path='snapshots/'+path.name,sha256=runner.sha(path))
            path.with_suffix('.json').write_bytes(runner.canonical(metadata))
            return {**self.p, 'snapshot':str(path), 'snapshot_sha256':runner.sha(path),
                    'snapshot_metadata_sha256':runner.sha(path.with_suffix('.json')),
                    'training_protocol_sha256':value['protocol_sha256']}
        valid = publish(saved,'valid')
        with mock.patch.object(core_model,'build_model',side_effect=fake_build) as builder:
            loaded, identity = runner.load_snapshot_model(valid)
            self.assertEqual(identity['parameter_counts']['total'],3)
            torch.testing.assert_close(loaded.weight,fake.weight,rtol=0,atol=0)
            self.assertEqual(builder.call_count,1)
            malformed = copy.deepcopy(saved)
            malformed['protocol']['model_config']['global_q_heads'] = 100000000
            malformed['protocol_sha256'] = hashlib.sha256(runner.canonical(malformed['protocol'])).hexdigest()
            bad = publish(malformed,'enormous-heads')
            with self.assertRaisesRegex(ValueError,'fixed HybridConfig'):
                runner.load_snapshot_model(bad)
            self.assertEqual(builder.call_count,1)
            with self.assertRaisesRegex(ValueError,'SHA256'):
                runner.load_snapshot_model({**valid,'snapshot_sha256':'0'*64})
            tampered = copy.deepcopy(saved)
            tampered['source_hashes'] = {}
            with self.assertRaisesRegex(ValueError,'source hashes'):
                runner.load_snapshot_model(publish(tampered,'missing-source'))
            broken_state = copy.deepcopy(saved)
            broken_state['model_state'] = {'other':torch.zeros(3)}
            with self.assertRaisesRegex(ValueError,'state keys'):
                runner.load_snapshot_model(publish(broken_state,'wrong-state'))

    def test_06_cooperative_wall_boundary_stops_before_model_load(self):
        path,digest = self.protocol({**self.p,'max_wall_seconds':1})
        with mock.patch.object(runner.time,'perf_counter',side_effect=[0,2,2]), \
             mock.patch.object(runner,'load_snapshot_model') as loader:
            with self.assertRaisesRegex(TimeoutError,'Cooperative'):
                runner.run_evaluation(path,digest,self.directory/'timeout',execute=True)
            loader.assert_not_called()
        failure=json.loads((self.directory/'timeout'/'failure.json').read_text(encoding='utf-8'))
        self.assertEqual(failure['counts']['model_forward_attempts'],0)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TaskRunnerTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'completed_utc':runner.utc(),'passed':result.wasSuccessful(),'tests_run':result.testsRun,
              'failures':[str(value) for value in result.failures],'errors':[str(value) for value in result.errors],
              'counts':COUNTS,'artifact_root':str(ARTIFACT_ROOT),
              'source_sha256':runner.sha(ROOT/'scripts/run_babylm_task_evaluation_v0.py'),
              'scope':'fake-only engineering runner gates, real reader and aggregator; zero real model forwards'}
    result_path=ROOT/'results/babylm-task-runner-fake-v0.json'
    result_path.write_bytes(runner.canonical(report)+b'\n')
    runner.append_json(ROOT/'results/babylm-task-runner-fake-v0-history.jsonl',report)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
