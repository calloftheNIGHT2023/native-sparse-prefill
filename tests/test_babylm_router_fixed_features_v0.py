"""Tiny synthetic CPU tests only; no corpus/checkpoint/cloud access.

Accounting for a successful run: 6 indexer forward-equivalent calls, 1 tiny attention
forward, 2 teacher-row projection calls (2 dense + 2 selected distributions),
3 backward calls and 1 router-only optimizer update. No LM model forwards.
"""
import dataclasses
import unittest

import torch

from scripts.run_babylm_router_fixed_features_v0 import (
    fresh_indexers, index_rows, topk_support, teacher_rows, support_kl,
    row_metrics, state_sha,
)
from src.babylm_hybrid.attention import GlobalAttention
from src.babylm_hybrid.config import HybridConfig
from src.babylm_hybrid.model import build_model


class FixedFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = dataclasses.replace(HybridConfig(), layer_types=("global", "global"))

    def test_fresh_initialization_matches_original_factory_and_preserves_rng(self):
        torch.manual_seed(321)
        before = torch.random.get_rng_state().clone()
        fresh = fresh_indexers(self.cfg, 123)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        model = build_model(self.cfg, "sparse", 42, 123)
        for original, got in zip(model.layers, fresh):
            self.assertEqual(state_sha(original.mixer.indexer.state_dict()), state_sha(got.state_dict()))

    def test_index_rows_matches_original_forward_and_gradient(self):
        a = fresh_indexers(self.cfg, 55)[0]
        b = fresh_indexers(self.cfg, 55)[0]
        hidden = torch.linspace(-2, 2, 32 * self.cfg.hidden_size).reshape(32, self.cfg.hidden_size)
        query = [15, 23, 30]
        old_scores, old_visible = a(hidden)
        new_scores, new_visible, _ = index_rows(b, hidden, query)
        self.assertTrue(torch.equal(old_visible[query], new_visible))
        self.assertTrue(torch.allclose(old_scores[query], new_scores, atol=2e-6, rtol=1e-6))
        old_scores[query].sum().backward()
        new_scores.sum().backward()
        for old, new in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.allclose(old.grad, new.grad, atol=2e-5, rtol=1e-5))

    def test_fixed_support_teacher_matches_existing_sparse_kl(self):
        torch.manual_seed(56)
        layer = GlobalAttention(self.cfg)
        layer.initialize_indexer()
        hidden = torch.randn(32, self.cfg.hidden_size)
        query = [15, 23, 30]
        score, visible, _ = index_rows(layer.indexer, hidden, query)
        fixed = topk_support(score, visible, self.cfg.selected_complete_blocks)
        with torch.no_grad():
            teacher = teacher_rows(layer, hidden, query, fixed)
            supervised = torch.zeros(32, dtype=torch.bool)
            supervised[query] = True
            _output, original_kl, _stats = layer._segment(hidden, "sparse", supervised)
        got = support_kl(score, teacher["target"], fixed).sum()
        self.assertAlmostEqual(float(got.detach()), float(original_kl), places=5)
        self.assertTrue(torch.allclose(teacher["target"].sum(-1), torch.ones(3), atol=1e-6))
        self.assertFalse(any(value.requires_grad for value in teacher.values()))

    def test_metric_uses_dynamic_topk_not_constant_cached_mask(self):
        visible = torch.ones(2, 8, dtype=torch.bool)
        first = torch.arange(8.).repeat(2, 1)
        second = first.flip(-1)
        fixed = topk_support(first, visible, 2)
        mass = torch.tensor([[.01, .01, .01, .01, .01, .01, .42, .42]]).repeat(2, 1)
        feature = {"fixed_support": fixed, "target": fixed.float() / 2,
                   "dense_mass": mass, "dense_target": mass / mass.sum(-1, keepdim=True),
                   "tail_mass": torch.full((2,), .10)}
        one = row_metrics(first, visible, first[:, None, :], feature, 2)
        two = row_metrics(second, visible, second[:, None, :], feature, 2)
        self.assertTrue(torch.equal(one["fixed_mask_dense_mass_coverage_control"], two["fixed_mask_dense_mass_coverage_control"]))
        self.assertTrue(bool((one["dynamic_topk_dense_mass_coverage"] > two["dynamic_topk_dense_mass_coverage"] + .5).all()))
        self.assertTrue(bool((one["oracle_minus_random_coverage"] > 0).all()))
        scaled = row_metrics(first / 8, visible, first[:, None, :], feature, 2, score_scale=1/8)
        self.assertTrue(torch.equal(one["fixed_support_kl_common_score_scale_1"], scaled["fixed_support_kl_common_score_scale_1"]))
        self.assertTrue(torch.equal(one["visible_score_span"] / 8, scaled["visible_score_span"]))

    def test_router_update_leaves_features_and_teacher_immutable(self):
        indexers = fresh_indexers(self.cfg, 101)
        layer = GlobalAttention(self.cfg)
        hidden = torch.linspace(-3, 2, 32 * self.cfg.hidden_size).reshape(32, self.cfg.hidden_size)
        query = [15, 23, 30]
        score, visible, _ = index_rows(indexers[0], hidden, query)
        fixed = topk_support(score, visible, 2)
        with torch.no_grad():
            target = teacher_rows(layer, hidden, query, fixed)["target"]
        before_hidden = hidden.clone()
        before_target = target.clone()
        initial = state_sha(indexers.state_dict())
        optimizer = torch.optim.AdamW(indexers.parameters(), lr=1e-3, betas=(.9, .95), weight_decay=.1, eps=1e-8)
        optimizer.zero_grad(set_to_none=True)
        # Deliberately update only one tiny indexer here, not a whole model.
        loss = support_kl(score, target, fixed).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(indexers.parameters(), 1.0)
        optimizer.step()
        self.assertNotEqual(initial, state_sha(indexers.state_dict()))
        self.assertTrue(torch.equal(hidden, before_hidden) and torch.equal(target, before_target))
        self.assertIsNone(hidden.grad)
        self.assertTrue(all(parameter.grad is None for parameter in layer.parameters()))

    def test_no_future_support_and_scale_preserves_initial_selection(self):
        indexer = fresh_indexers(self.cfg, 404)[0]
        hidden = torch.arange(1024.).reshape(32, 32) / 1024
        score, visible, _ = index_rows(indexer, hidden, [10, 16, 30])
        selected = topk_support(score, visible, 2)
        self.assertFalse(bool((selected & ~visible).any()))
        self.assertTrue(torch.equal(selected.sum(-1), torch.tensor([2, 2, 2])))
        self.assertTrue(torch.equal(selected, topk_support(score / 8, visible, 2)))


if __name__ == "__main__":
    unittest.main()
