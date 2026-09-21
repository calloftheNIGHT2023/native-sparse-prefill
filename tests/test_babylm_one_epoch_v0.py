"""CPU-only boundary and clipping regressions; no scientific training data."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from src.babylm_hybrid import evaluation, training
from src.babylm_hybrid.model import HybridLM
from tests import test_babylm_training_v0 as fixtures


ROOT = Path(__file__).resolve().parents[1]


class OneEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        cls.root = ROOT / 'logs' / ('babylm-one-epoch-engineering-' + stamp)
        cls.root.mkdir(parents=True)
        cls.counts = {'model_forward_calls': 0, 'backward_calls': 0,
                      'engineering_optimizer_steps': 0, 'scientific_optimizer_steps': 0,
                      'real_training_tokens': 0, 'gpu_calls': 0}
        forward, backward, step = HybridLM.forward, torch.Tensor.backward, torch.optim.AdamW.step

        def counted_forward(model, *args, **kwargs):
            cls.counts['model_forward_calls'] += 1
            return forward(model, *args, **kwargs)

        def counted_backward(tensor, *args, **kwargs):
            cls.counts['backward_calls'] += 1
            return backward(tensor, *args, **kwargs)

        def counted_step(optimizer, *args, **kwargs):
            cls.counts['engineering_optimizer_steps'] += 1
            return step(optimizer, *args, **kwargs)

        cls.patches = [mock.patch.object(HybridLM, 'forward', counted_forward),
                       mock.patch.object(torch.Tensor, 'backward', counted_backward),
                       mock.patch.object(torch.optim.AdamW, 'step', counted_step)]
        for patch in cls.patches:
            patch.start()

    @classmethod
    def tearDownClass(cls):
        for patch in reversed(cls.patches):
            patch.stop()
        (cls.root / 'engineering-counts.json').write_text(json.dumps(cls.counts, indent=2), encoding='utf-8')
        print('ENGINEERING_ARTIFACTS=' + str(cls.root))
        print('ENGINEERING_COUNTS=' + json.dumps(cls.counts, sort_keys=True))

    def setUp(self):
        self.directory = self.root / self._testMethodName
        self.directory.mkdir()

    def dataset(self, specs):
        return fixtures.SyntheticWindowDataset(self.directory / 'fixture', specs)

    def protocol(self, dataset, **changes):
        result = fixtures.TrainingEngineTests.protocol(self, dataset)
        result.update(max_epochs=1, max_updates=10, windows_per_update=16,
                      max_word_exposures=sum(row['word_exposures'] for row in dataset.rows),
                      gradient_clip_scope='separate_backbone_indexer')
        result.update(changes)
        return result

    def run_engine(self, protocol, mode, name, dataset, **kwargs):
        directory = self.directory / name
        with mock.patch.object(training, 'load_window_dataset', return_value=dataset):
            result = training.train_run(protocol, mode, directory, **kwargs)
        return result, directory

    def test_exact_epoch_partial_six_tail_and_paired_lr(self):
        dataset = self.dataset([(8 + i % 3, 2 + i % 2) for i in range(22)])
        protocol = self.protocol(dataset, learning_rate=3e-4, indexer_learning_rate=1e-3,
                                 warmup_word_exposures=5, min_lr_ratio=0.1)
        original_order = training.epoch_permutation

        def no_second_epoch(seed, epoch, length):
            self.assertEqual(epoch, 0, 'The next epoch must not even be planned')
            return original_order(seed, epoch, length)

        results = []
        with mock.patch.object(training, 'epoch_permutation', side_effect=no_second_epoch):
            for mode in ('dense', 'sparse'):
                with mock.patch.object(dataset, 'window', wraps=dataset.window) as reader:
                    summary, directory = self.run_engine(protocol, mode, mode, dataset)
                self.assertEqual(reader.call_count, 22)
                updates = [e for e in fixtures.read_events(directory) if e['type'] == 'update']
                self.assertEqual([len(e['window_ids']) for e in updates], [16, 6])
                ids = [item['window_index'] for row in updates for item in row['window_ids']]
                self.assertEqual(sorted(ids), list(range(22)))
                self.assertEqual(summary['status'], 'epoch_complete')
                self.assertEqual(summary['cursor'], {'epoch': 1, 'position': 0})
                for name, expected in [('windows', 22), ('updates', 2), ('forward_calls', 22),
                                       ('backward_calls', 22), ('word_exposures', 55),
                                       ('input_tokens', sum(x['input_tokens'] for x in dataset.rows)),
                                       ('loss_tokens', sum(x['loss_tokens'] for x in dataset.rows))]:
                    self.assertEqual(summary['counts'][name], expected)
                self.assertAlmostEqual(summary['last_lr']['backbone'], 3e-5, places=12)
                self.assertTrue(summary['snapshot_state']['final_receipts'][-1]['protocol_completion_boundary'])
                self.assertTrue(all(e['gradient_clip_scope'] == 'separate_backbone_indexer' for e in updates))
                results.append(updates)
        for d, e in zip(*results):
            for key in ('window_ids', 'counts', 'lr_word_position', 'lr_multiplier'):
                self.assertEqual(d[key], e[key])
            self.assertEqual(d['lr']['backbone'], e['lr']['backbone'])

    def test_short_word_cap_cannot_claim_complete_epoch(self):
        dataset = self.dataset([(8, 2)] * 5)
        summary, directory = self.run_engine(self.protocol(dataset, max_word_exposures=5), 'dense', 'cap', dataset)
        self.assertEqual(summary['status'], 'word_budget_reached')
        self.assertEqual(summary['counts']['windows'], 2)
        self.assertEqual(summary['cursor']['epoch'], 0)
        self.assertFalse(summary['snapshot_state']['final_receipts'][-1]['protocol_completion_boundary'])

    def test_update_cap_cannot_claim_complete_epoch(self):
        dataset = self.dataset([(8, 2)] * 5)
        summary, _ = self.run_engine(self.protocol(dataset, windows_per_update=2, max_updates=1), 'dense', 'cap', dataset)
        self.assertEqual(summary['status'], 'max_updates_reached')
        self.assertEqual(summary['counts']['windows'], 2)
        self.assertFalse(summary['snapshot_state']['final_receipts'][-1]['protocol_completion_boundary'])

    def test_omitted_epoch_and_clip_fields_keep_legacy_behavior(self):
        dataset = self.dataset([(8, 2)] * 3)
        protocol = self.protocol(dataset, windows_per_update=2, max_updates=2, max_word_exposures=100)
        del protocol['max_epochs']
        del protocol['gradient_clip_scope']
        summary, directory = self.run_engine(protocol, 'sparse', 'legacy', dataset)
        self.assertEqual(summary['status'], 'max_updates_reached')
        self.assertEqual(summary['counts']['windows'], 4)
        self.assertEqual(summary['cursor'], {'epoch': 1, 'position': 1})
        updates = [e for e in fixtures.read_events(directory) if e['type'] == 'update']
        self.assertTrue(all(e['gradient_scope'] == 'combined_objective_before_global_clipping' for e in updates))

    def test_resume_partial_tail_matches_continuous_and_finished_resume_is_noop(self):
        dataset = self.dataset([(8, 2), (12, 3), (16, 4), (20, 5), (24, 6)])
        protocol = self.protocol(dataset, windows_per_update=2)
        continuous, full_dir = self.run_engine(protocol, 'sparse', 'full', dataset)
        partial, resume_dir = self.run_engine(protocol, 'sparse', 'resume', dataset, stop_after_updates=2)
        self.assertEqual(partial['status'], 'stopped_by_update_limit')
        resumed, _ = self.run_engine(protocol, 'sparse', 'resume', dataset, resume_checkpoint=resume_dir / 'checkpoint.pt')
        self.assertEqual(resumed['status'], 'epoch_complete')
        full = torch.load(full_dir / 'checkpoint.pt', map_location='cpu', weights_only=False)
        resumed_state = torch.load(resume_dir / 'checkpoint.pt', map_location='cpu', weights_only=False)
        for name in ('model_state', 'optimizer_state', 'counters', 'cursor', 'python_rng', 'numpy_rng', 'torch_rng'):
            fixtures.assert_nested_equal(self, full[name], resumed_state[name])
        with mock.patch.object(dataset, 'window', side_effect=AssertionError('Finished epoch must not read data')):
            finished, _ = self.run_engine(protocol, 'sparse', 'resume', dataset,
                                           resume_checkpoint=resume_dir / 'checkpoint.pt')
        self.assertEqual(finished['status'], 'epoch_complete')
        self.assertEqual(finished['counts'], continuous['counts'])
        self.assertEqual(len(finished['snapshot_state']['final_receipts']), 2)

    def test_final_eval_runs_once_after_epoch_and_zero_target_window_is_consumed(self):
        dataset = self.dataset([(8, 2), (1, 1), (12, 3)])
        manifest = self.directory / 'eval-manifest.json'
        manifest.write_text('{"synthetic_evaluator_stub":true}', encoding='utf-8')
        protocol = self.protocol(dataset, eval_manifest=str(manifest), eval_manifest_sha256=fixtures.sha(manifest),
                                 eval_window_indices=[0], eval_final=True)
        metrics = {'total': {'word_exposures': 2, 'input_tokens': 8, 'loss_tokens': 7, 'forward_calls': 1}}
        with mock.patch.object(evaluation, 'evaluate_windows', return_value=copy.deepcopy(metrics)) as evaluator:
            summary, directory = self.run_engine(protocol, 'dense', 'run', dataset)
            self.assertEqual(evaluator.call_count, 1)
            resumed, _ = self.run_engine(protocol, 'dense', 'run', dataset, resume_checkpoint=directory / 'checkpoint.pt')
            self.assertEqual(evaluator.call_count, 1)
        self.assertEqual(summary['counts']['windows'], 3)
        self.assertEqual(summary['counts']['forward_calls'], 2)
        self.assertEqual(summary['counts']['skipped_zero_target_input_tokens'], 1)
        self.assertEqual(resumed['eval_counts']['evaluations'], 1)

    def test_invalid_epoch_and_clip_scope_rejected_before_model_compute(self):
        dataset = self.dataset([(8, 2)])
        counts = dict(self.counts)
        for bad in (0, -1, True, 1.5, None):
            with self.subTest(max_epochs=bad), self.assertRaises(ValueError):
                training._validate_protocol(self.protocol(dataset, max_epochs=bad))
        with self.assertRaises(ValueError):
            training._validate_protocol(self.protocol(dataset, gradient_clip_scope='per_tensor'))
        self.assertEqual(self.counts, counts)

    def test_large_indexer_gradient_cannot_clip_backbone_in_separate_scope(self):
        def clipped(index_magnitude, scope):
            model = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))])
            model[0].grad = torch.tensor([3.0, 4.0])
            model[1].grad = torch.tensor([index_magnitude])
            norm, reported, main, index = training._clip_parameter_groups(model, [model[0]], [model[1]], 1.0, scope)
            return model[0].grad.clone(), model[1].grad.clone(), float(norm), reported, main, index

        small = clipped(0.1, 'separate_backbone_indexer')
        large = clipped(1000.0, 'separate_backbone_indexer')
        torch.testing.assert_close(small[0], large[0], rtol=0, atol=0)
        torch.testing.assert_close(large[0], torch.tensor([0.6, 0.8]), rtol=1e-6, atol=1e-6)
        self.assertEqual(large[3], large[4])
        self.assertAlmostEqual(large[4], 1 / (5 + 1e-6), places=7)
        self.assertAlmostEqual(large[5], 1 / (1000 + 1e-6), places=7)
        self.assertAlmostEqual(large[2], (25 + 1000**2)**0.5, places=4)
        self.assertGreater(float(large[0].norm()), float(clipped(1000.0, 'global')[0].norm()) * 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
