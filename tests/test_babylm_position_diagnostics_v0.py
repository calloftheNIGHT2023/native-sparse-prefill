"""Fixed-position loss diagnostics, checked against a separate scalar oracle."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.babylm_hybrid.evaluation import evaluate_windows

COUNTS = {"fake_model_forward_calls": 0, "whole_model_forward_calls": 0,
          "backward_calls": 0, "optimizer_updates": 0, "gpu_calls": 0}


class LogitLM(nn.Module):
    def __init__(self, bad_loss=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.child = nn.Dropout(.1)
        self.bad_loss = bad_loss

    @staticmethod
    def logits(length):
        positions = torch.arange(length, dtype=torch.float64)[:, None]
        vocab = torch.arange(7, dtype=torch.float64)[None, :]
        return torch.cos(positions * .123 + vocab * .4).unsqueeze(0)

    def forward(self, ids, aux_weight=0):
        COUNTS["fake_model_forward_calls"] += 1
        assert not self.training and not self.child.training
        assert not torch.is_grad_enabled() and aux_weight == 0
        logits = self.logits(ids.shape[1])
        loss = (torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
                if ids.shape[1] > 1 else torch.tensor(float('nan')))
        if self.bad_loss:
            loss = loss + 1
        return SimpleNamespace(logits=logits, lm_loss=loss,
                               token_loss_count=ids.shape[1]-1, attention_stats=[])


class PositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'manifest.json'
        self.items = []
        for idx, length in enumerate([1026, 8, 1]):
            self.items.append(dict(window_index=idx, source_index=idx % 2,
                input_ids=np.asarray([i % 7 for i in range(length)], dtype=np.uint32),
                input_tokens=length, loss_tokens=length-1, word_exposures=length,
                single_segment=True, reset_model_state_before=True))
        self.path.write_text(json.dumps(dict(total_windows=3, source_summaries=[
            dict(source_index=0, source='a'), dict(source_index=1, source='b')])))
        self.reader = patch('src.babylm_hybrid.evaluation.iter_windows',
            side_effect=lambda path, indices: iter(self.items[i] for i in indices))
        self.reader.start()

    def tearDown(self):
        self.reader.stop()
        self.temp.cleanup()

    def test_position_and_length_partitions_match_scalar_shifted_oracle(self):
        before = COUNTS['fake_model_forward_calls']
        model = LogitLM().train()
        model.child.eval()
        result = evaluate_windows(model, self.path, [0, 1, 2], position_diagnostics=True)
        self.assertEqual(COUNTS['fake_model_forward_calls'] - before, 3)
        diag = result['position_diagnostics']
        self.assertEqual(diag['extra_model_forwards'], 0)
        expected = {k: [] for k in ['1-256', '257-512', '513-1024', '1025-2048']}
        for row in self.items:
            raw = LogitLM.logits(row['input_tokens'])[0].tolist()
            for i in range(row['loss_tokens']):
                scores = raw[i]
                value = math.log(math.fsum(math.exp(x) for x in scores)) - scores[int(row['input_ids'][i+1])]
                label = ('1-256' if i < 256 else '257-512' if i < 512 else
                         '513-1024' if i < 1024 else '1025-2048')
                expected[label].append(value)
        for key, values in expected.items():
            actual = diag['query_history_bins'][key]
            self.assertEqual(actual['loss_tokens'], len(values))
            self.assertAlmostEqual(actual['nll_sum'], math.fsum(values), places=10)
        self.assertEqual([diag['query_history_bins'][x]['loss_tokens'] for x in expected], [263,256,512,1])
        self.assertEqual(sum(v['loss_tokens'] for v in diag['window_length_bins'].values()), 1032)
        self.assertAlmostEqual(sum(v['nll_sum'] for v in diag['window_length_bins'].values()),
                               result['total']['nll_sum'], places=10)
        self.assertEqual(diag['window_length_bins']['1-256']['windows'], 2)
        self.assertEqual(diag['per_source_query_history_bins']['b']['1-256']['loss_tokens'], 7)
        self.assertIsNone(diag['per_window'][2]['nll'])
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        self.assertIsNone(model.weight.grad)

    def test_inconsistent_logits_and_declared_loss_rejected(self):
        model = LogitLM(bad_loss=True).train()
        with self.assertRaisesRegex(ValueError, 'logits/LM-loss mismatch'):
            evaluate_windows(model, self.path, [1], position_diagnostics=True)
        self.assertTrue(model.training)

    def test_empty_selection_has_no_fake_calls_and_null_nll(self):
        before = COUNTS['fake_model_forward_calls']
        result = evaluate_windows(LogitLM(), self.path, [], position_diagnostics=True)
        self.assertEqual(COUNTS['fake_model_forward_calls'], before)
        self.assertEqual(result['position_diagnostics']['per_window'], [])
        self.assertTrue(all(v['nll'] is None for v in result['position_diagnostics']['query_history_bins'].values()))

    def test_bad_flag_and_out_of_scope_length_rejected_before_forward(self):
        before = COUNTS['fake_model_forward_calls']
        with self.assertRaisesRegex(ValueError, 'must be a bool'):
            evaluate_windows(LogitLM(), self.path, [0], position_diagnostics=1)
        self.items[0].update(input_ids=np.asarray([i % 7 for i in range(2049)], dtype=np.uint32), input_tokens=2049, loss_tokens=2048)
        with self.assertRaisesRegex(ValueError, 'at most 2048'):
            evaluate_windows(LogitLM(), self.path, [0], position_diagnostics=True)
        self.assertEqual(COUNTS['fake_model_forward_calls'], before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
