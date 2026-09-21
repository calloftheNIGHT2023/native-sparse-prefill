"""CPU-only attention-component tests; no full LM, backwards, or updates."""
import json
import math
import unittest
from dataclasses import replace

import torch
from torch import nn

from src.babylm_hybrid.attention import GlobalAttention
from src.babylm_hybrid.config import HybridConfig
from src.babylm_hybrid.routing_diagnostics import POLICIES, RoutingDiagnostics, select_support


COUNTS = {"full_lm_forwards": 0, "attention_component_forwards": 0,
          "diagnostic_segment_observations": 0, "backwards": 0, "optimizer_updates": 0}


def attention_call(layer, x, segment_ids, loss_mask=None):
    COUNTS["attention_component_forwards"] += 1
    with torch.no_grad():
        return layer(x, segment_ids, mode="sparse", loss_mask=loss_mask)


class RoutingDiagnosticTests(unittest.TestCase):
    def make_layer(self, block_size=2, topk=2):
        cfg = replace(HybridConfig(), hidden_size=8, global_q_heads=2, global_kv_heads=1,
                      global_head_dim=4, global_rotary_dim=2, index_query_heads=2,
                      index_head_dim=4, index_rotary_dim=2, block_size=block_size,
                      selected_complete_blocks=topk)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(720)
            layer = GlobalAttention(cfg).double()
            layer.initialize_indexer()
        return layer

    def input(self, length):
        rng = torch.Generator().manual_seed(19)
        return torch.randn(1, length, 8, dtype=torch.float64, generator=rng)

    def add_observations(self, d):
        COUNTS["diagnostic_segment_observations"] += d.forward_counts["observed_sparse_segment_forwards"]

    def test_support_matches_reference_and_causal_budget_at_boundaries(self):
        layer = self.make_layer()
        for length in (1, 2, 3, 4, 5, 6, 7, 12, 13):
            scores = torch.zeros(length, length // 2, dtype=torch.float64)
            positions = torch.arange(length)
            visible = (torch.arange(length // 2)[None, :] + 1) * 2 - 1 <= positions[:, None]
            original_chosen, original_mask = layer._sparse_selection(scores, visible)
            chosen, mask = select_support(scores, visible, block_size=2, topk=2)
            self.assertTrue(torch.equal(chosen, original_chosen))
            self.assertTrue(torch.equal(mask, original_mask))
            for policy in POLICIES:
                chosen, mask = select_support(scores, visible, block_size=2, topk=2,
                                              policy=policy, seed=33)
                self.assertTrue(torch.equal(chosen.sum(-1), visible.sum(-1).clamp_max(2)))
                self.assertFalse((chosen & ~visible).any())
                self.assertFalse((mask & torch.ones_like(mask).triu(1)).any())
                expected_tokens = chosen.sum(-1) * 2 + (positions + 1) % 2
                self.assertTrue(torch.equal(mask.sum(-1), expected_tokens))
                for query in range(length):
                    tail_start = ((query + 1) // 2) * 2
                    self.assertTrue(mask[query, tail_start:query + 1].all())

    def test_random_replay_global_rng_and_call_order(self):
        length = 25
        scores = torch.randn(length, length // 2, generator=torch.Generator().manual_seed(1))
        visible = (torch.arange(length // 2)[None, :] + 1) * 2 - 1 <= torch.arange(length)[:, None]
        before = torch.get_rng_state().clone()
        first = select_support(scores, visible, block_size=2, topk=2, policy="random", seed=21, layer_name="a")
        select_support(scores, visible, block_size=2, topk=2, policy="prefix", seed=21, layer_name="a")
        replay = select_support(scores + 500, visible, block_size=2, topk=2, policy="random", seed=21, layer_name="a")
        other = select_support(scores, visible, block_size=2, topk=2, policy="random", seed=22, layer_name="a")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(first[0], replay[0]))
        self.assertFalse(torch.equal(first[0], other[0]))

    def test_learned_is_exactly_original_and_restores_methods_parameters_flags(self):
        layer = self.make_layer()
        layer.train()
        x = self.input(13)
        segments = torch.zeros(1, 13, dtype=torch.long)
        supervision = torch.ones(1, 13, dtype=torch.bool)
        supervision[:, -1] = False
        before = {k: v.clone() for k, v in layer.state_dict().items()}
        baseline = attention_call(layer, x, segments, supervision)
        rng = torch.get_rng_state().clone()
        with RoutingDiagnostics(layer, policy="learned", position_edges=(4, 8)) as d:
            observed = attention_call(layer, x, segments, supervision)
        self.add_observations(d)
        for i in (0, 1):
            torch.testing.assert_close(observed[i], baseline[i], rtol=0, atol=0)
        self.assertEqual(observed[2], baseline[2])
        self.assertNotIn("_segment", layer.__dict__)
        self.assertNotIn("_sparse_selection", layer.__dict__)
        self.assertTrue(layer.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in layer.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))
        report = d.summary()
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["forward_counts"]["extra_counterfactual_policy_selections"], 4)
        self.assertEqual(report["forward_counts"]["original_sparse_selection_calls"], 1)
        self.assertEqual(report["forward_counts"]["additional_full_lm_forwards"], 0)
        nodes = report["layers"]["<root>"]
        self.assertEqual(nodes["all_queries"]["all"]["effective_topk"]["query_count"], 8)
        self.assertEqual(nodes["supervised_queries"]["all"]["effective_topk"]["query_count"], 7)

    def test_uniform_main_attention_coverage_is_retained_fraction(self):
        layer = self.make_layer()
        with torch.no_grad():
            layer.q_proj.weight.zero_()
            layer.k_proj.weight.zero_()
            layer.indexer.q_proj.weight.zero_()
        length = 13
        segments = torch.zeros(1, length, dtype=torch.long)
        loss_mask = torch.ones_like(segments, dtype=torch.bool)
        loss_mask[:, -1] = False
        with RoutingDiagnostics(layer, position_edges=(4, 8)) as d:
            attention_call(layer, self.input(length), segments, loss_mask)
        self.add_observations(d)
        node = d.summary()["layers"]["<root>"]["all_queries"]["all"]["effective_topk"]
        expected = [(4 + p % 2) / p for p in range(6, length + 1)]
        for policy in POLICIES:
            self.assertAlmostEqual(node["metrics"][f"coverage_{policy}"]["mean"], sum(expected) / len(expected), places=13)
        metrics = node["metrics"]
        self.assertEqual(metrics["zero_score_query"]["sum"], 8)
        self.assertEqual(metrics["zero_score_query"]["count"], 8)
        self.assertEqual(metrics["all_pre_relu_nonpositive_query"]["sum"], 8)
        self.assertEqual(metrics["all_pre_relu_strict_negative_query"]["sum"], 0)
        self.assertEqual(metrics["learned_prefix_complete_block_overlap"]["mean"], 1)
        self.assertAlmostEqual(metrics["learned_selected_score_entropy_nats"]["mean"], math.log(2), places=13)

    def test_coverage_matches_independent_full_attention_sum(self):
        layer = self.make_layer()
        x = self.input(11)
        with torch.no_grad():
            q, k, _ = layer._main_qkv(x[0])
            logits = torch.einsum("thd,shd->ths", q, k) / math.sqrt(layer.head_dim)
            p = logits.masked_fill(~torch.ones(11, 11, dtype=torch.bool).tril()[:, None, :], -torch.inf).softmax(-1).mean(1)
            scores, visible = layer.indexer(x[0])
            choices = {}
            for policy in POLICIES:
                _, mask = select_support(scores, visible, block_size=2, topk=2, policy=policy, seed=6, layer_name="<root>")
                choices[policy] = (p * mask).sum(-1)
        with RoutingDiagnostics(layer, seed=6) as d:
            attention_call(layer, x, torch.zeros(1, 11, dtype=torch.long))
        self.add_observations(d)
        metrics = d.summary()["layers"]["<root>"]["all_queries"]["all"]["effective_topk"]["metrics"]
        for policy in POLICIES:
            self.assertAlmostEqual(metrics[f"coverage_{policy}"]["mean"], float(choices[policy][5:].mean()), places=13)

    def test_no_complete_blocks_and_segment_boundaries(self):
        layer = self.make_layer(block_size=4)
        x = self.input(9)
        segments = torch.tensor([[0, 0, 0, -1, 1, 1, 1, 1, 1]])
        supervised = torch.tensor([[True, True, False, False, True, True, True, True, False]])
        with RoutingDiagnostics(layer, position_edges=(2, 4)) as d:
            attention_call(layer, x, segments, supervised)
        self.add_observations(d)
        report = d.summary()
        json.dumps(report, allow_nan=False)
        nodes = report["layers"]["<root>"]
        self.assertEqual(nodes["all_queries"]["all"]["all"]["query_count"], 8)
        self.assertEqual(nodes["supervised_queries"]["all"]["all"]["query_count"], 6)
        self.assertEqual(nodes["all_queries"]["all"]["nonempty_visible"]["query_count"], 2)
        self.assertEqual(nodes["supervised_queries"]["all"]["nonempty_visible"]["query_count"], 1)
        self.assertEqual(nodes["all_queries"]["all"]["effective_topk"]["query_count"], 0)
        for policy in POLICIES:
            self.assertAlmostEqual(nodes["all_queries"]["all"]["all"]["metrics"][f"coverage_{policy}"]["mean"], 1., places=13)
        self.assertEqual(report["forward_counts"]["observed_sparse_segment_forwards"], 2)

    def test_exception_restoration_and_nested_rejection(self):
        layer = self.make_layer()
        original = layer._segment
        # Instance-owned method must be restored rather than deleted.
        layer._segment = original
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with RoutingDiagnostics(layer) as outer:
                with self.assertRaisesRegex(RuntimeError, "already has active"):
                    with RoutingDiagnostics(layer):
                        pass
                raise RuntimeError("deliberate")
        self.assertIs(layer.__dict__["_segment"], original)
        self.assertNotIn("_sparse_selection", layer.__dict__)
        with RoutingDiagnostics(layer):
            pass
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            with outer:
                pass

    def test_interventions_match_direct_support_and_keep_component_stats(self):
        layer = self.make_layer()
        x = self.input(17)
        scores, visible = layer.indexer(x[0])
        for policy in POLICIES:
            expected = select_support(scores, visible, block_size=2, topk=2, policy=policy, seed=11, layer_name="<root>")
            with RoutingDiagnostics(layer, policy=policy, seed=11) as d:
                actual = layer._sparse_selection(scores, visible)
                result = attention_call(layer, x, torch.zeros(1, 17, dtype=torch.long))
            self.add_observations(d)
            self.assertTrue(torch.equal(actual[0], expected[0]))
            self.assertTrue(torch.equal(actual[1], expected[1]))
            self.assertEqual(result[2]["logical_kept_pairs"], int(expected[1].sum()))

    def test_bad_inputs_fail_closed(self):
        layer = self.make_layer()
        with self.assertRaises(ValueError):
            RoutingDiagnostics(layer, policy="future")
        with self.assertRaises(ValueError):
            RoutingDiagnostics(layer, position_edges=(8, 4))
        with self.assertRaises(ValueError):
            RoutingDiagnostics(nn.Linear(8, 8))
        no_indexer = GlobalAttention(layer.cfg)
        with self.assertRaises(ValueError):
            RoutingDiagnostics(no_indexer)
        with self.assertRaises(ValueError):
            select_support(torch.zeros(4, 2), torch.ones(4, 2, dtype=torch.bool), block_size=2, topk=2)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        print("TINY_CPU_TEST_COUNTS=" + json.dumps(COUNTS, sort_keys=True))
