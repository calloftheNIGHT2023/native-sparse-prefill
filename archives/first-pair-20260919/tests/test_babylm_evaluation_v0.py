"""CPU-only synthetic evaluation accounting tests; no training or real models."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts.prepare_babylm_dev_windows_v0 import select_fast_panel
from src.babylm_hybrid.evaluation import evaluate_windows

COUNTS = {"fake_model_forward_calls": 0, "scientific_model_forward_calls": 0,
          "backward_calls": 0, "optimizer_updates": 0, "gpu_calls": 0}


class FakeLM(nn.Module):
    def __init__(self, count_error=False, nan_nonempty=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(7.0))
        self.child = nn.Dropout(.5)
        self.count_error = count_error
        self.nan_nonempty = nan_nonempty
        self.seen_modes = []

    def forward(self, inputs, aux_weight=0.0):
        COUNTS["fake_model_forward_calls"] += 1
        assert not torch.is_grad_enabled()
        assert not self.training and not self.child.training
        assert aux_weight == 0.0
        self.seen_modes.append((self.training, self.child.training))
        length = inputs.shape[1]
        count = length - 1
        value = float(inputs[0, 0]) if count and not self.nan_nonempty else float("nan")
        return SimpleNamespace(lm_loss=torch.tensor(value), loss=torch.tensor(9999.),
                               token_loss_count=count + int(self.count_error),
                               attention_stats=[{"logical_dense_causal_pairs": length * (length + 1) // 2,
                                                 "logical_kept_pairs": length, "segments": 1}])


def item(idx, source, ids, words):
    ids = np.asarray(ids, dtype=np.uint32)
    ids.flags.writeable = False
    return {"window_index": idx, "source_index": source, "input_ids": ids,
            "word_exposures": words, "input_tokens": len(ids), "loss_tokens": len(ids) - 1,
            "single_segment": True, "reset_model_state_before": True}


class EvaluationAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "manifest.json"
        self.path.write_text(json.dumps({"total_windows": 3,
            "source_summaries": [{"source_index": 0, "source": "a.dev"},
                                 {"source_index": 1, "source": "b.dev"}]}), encoding="utf-8")
        self.items = [item(0, 0, [2, 9], 2), item(1, 0, [4, 7, 8, 9], 3), item(2, 1, [5], 1)]
        self.reader = patch("src.babylm_hybrid.evaluation.iter_windows",
                            side_effect=lambda path, indices: iter(self.items[i] for i in indices))
        self.reader.start()

    def tearDown(self):
        self.reader.stop()
        self.temp.cleanup()

    def test_01_token_weighted_nll_counts_and_mixed_mode_restoration(self):
        model = FakeLM().train()
        model.child.eval()
        before = model.weight.detach().clone()
        result = evaluate_windows(model, self.path, [0, 1, 2])
        self.assertEqual(result["total"]["nll_sum"], 14.0)
        self.assertEqual(result["total"]["nll"], 3.5)  # (2*1 + 4*3)/4, not (2+4)/2.
        for key, expected in [("windows", 3), ("forward_calls", 3), ("input_tokens", 7),
                              ("word_exposures", 6), ("loss_tokens", 4), ("zero_target_windows", 1)]:
            self.assertEqual(result["total"][key], expected)
        self.assertEqual(result["per_source"]["a.dev"]["nll"], 3.5)
        self.assertIsNone(result["per_source"]["b.dev"]["nll"])
        self.assertEqual(result["total"]["attention_counts"]["logical_dense_causal_pairs"], 14)
        self.assertEqual(result["total"]["logical_retained_fraction"], 7 / 14)
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        self.assertTrue(result["model_training_mode_restored"])
        torch.testing.assert_close(model.weight, before)
        self.assertIsNone(model.weight.grad)

    def test_02_empty_targets_nan_loss_is_not_multiplied_by_zero(self):
        result = evaluate_windows(FakeLM().eval(), self.path, [2])
        self.assertIsNone(result["total"]["nll"])
        self.assertEqual(result["total"]["nll_sum"], 0.)
        self.assertEqual(result["total"]["forward_calls"], 1)
        self.assertEqual(result["total"]["input_tokens"], 1)
        self.assertEqual(result["total"]["word_exposures"], 1)

    def test_03_max_windows_and_empty_selection(self):
        model = FakeLM().eval()
        result = evaluate_windows(model, self.path, None, max_windows=1)
        self.assertEqual(result["window_indices"], [0])
        self.assertEqual(result["total"]["nll"], 2.)
        self.assertFalse(model.training)
        before = COUNTS["fake_model_forward_calls"]
        result = evaluate_windows(model, self.path, [0, 1], max_windows=0)
        self.assertEqual(result["total"]["forward_calls"], 0)
        self.assertIsNone(result["total"]["nll"])
        self.assertEqual(COUNTS["fake_model_forward_calls"], before)

    def test_04_count_mismatch_restores_mode_after_exception(self):
        model = FakeLM(count_error=True).train()
        model.child.eval()
        with self.assertRaisesRegex(ValueError, "loss-token mismatch"):
            evaluate_windows(model, self.path, [0])
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)

    def test_05_nonfinite_nonempty_loss_rejected_and_mode_restored(self):
        model = FakeLM(nan_nonempty=True).train()
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            evaluate_windows(model, self.path, [0])
        self.assertTrue(model.training)
        self.assertTrue(model.child.training)

    def test_06_invalid_index_and_cap_are_not_silently_accepted(self):
        model = FakeLM()
        before = COUNTS["fake_model_forward_calls"]
        for indices, cap in [([0, 0], None), ([3], None), ([-1], None), ([True], None), ([0], -1)]:
            with self.assertRaises(ValueError):
                evaluate_windows(model, self.path, indices, max_windows=cap)
        self.assertEqual(COUNTS["fake_model_forward_calls"], before)

    def test_07_panel_hash_selection_is_independent_of_model_and_length_order(self):
        rows = []
        for source in [0, 1]:
            for i in range(11):
                start, length = i * 100, 5 + i
                rows.append([source, i, start, start + length, 0, 0, 0, length,
                             2 + i, length, length - 1])
        arr = np.asarray(rows, dtype=np.uint64)
        sources = [{"source_index": 0, "source": "a.dev"}, {"source_index": 1, "source": "b.dev"}]
        panel = select_fast_panel(arr, sources, per_source=8)
        self.assertEqual(panel["selected_windows"], 16)
        expected = []
        for sid, name in [(0, "a.dev"), (1, "b.dev")]:
            ranked = [(hashlib.sha256(f"babylm-dev-panel-v0|{name}|{i*100}|{i*100+5+i}".encode()).hexdigest(), sid * 11 + i)
                      for i in range(11)]
            expected.extend(idx for _, idx in sorted(ranked)[:8])
        self.assertEqual(panel["window_indices"], expected)
        self.assertEqual(panel["loss_tokens"], sum(int(arr[i, 10]) for i in expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
