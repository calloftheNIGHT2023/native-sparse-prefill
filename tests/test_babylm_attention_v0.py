"""CPU mathematical tests only; no training loops or optimizer updates."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from babylm_hybrid.attention import GlobalAttention, _rope


def config(**updates):
    values = dict(hidden_size=24, global_q_heads=4, global_kv_heads=2, global_head_dim=8,
                  global_rotary_dim=4, rope_theta=1e7, norm_eps=1e-6,
                  index_query_heads=4, index_head_dim=16, index_rotary_dim=8,
                  block_size=4, selected_complete_blocks=2, index_score_scale=1.0)
    values.update(updates)
    return SimpleNamespace(**values)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(170918)
        torch.set_num_threads(2)

    def build(self, **updates):
        attention = GlobalAttention(config(**updates)).double()
        with torch.random.fork_rng():
            torch.manual_seed(91)
            attention.initialize_indexer()
        return attention

    def test_delayed_indexer_and_dense_does_not_call_it(self):
        module = GlobalAttention(config()).double()
        self.assertIsNone(module.indexer)
        x = torch.randn(1, 7, 24, dtype=torch.float64)
        segments = torch.zeros(1, 7, dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, 'initialize_indexer'):
            module(x, segments, mode='sparse')
        module.initialize_indexer()
        self.assertEqual(next(module.indexer.parameters()).dtype, torch.float64)
        with self.assertRaisesRegex(RuntimeError, 'already initialized'):
            module.initialize_indexer()
        calls = []
        handle = module.indexer.register_forward_hook(lambda *args: calls.append(True))
        out, aux, stats = module(x, segments)
        handle.remove()
        self.assertFalse(calls)
        self.assertEqual(float(aux), 0)
        self.assertEqual(stats['indexer_score_elements'], 0)
        self.assertEqual(out.shape, x.shape)

    def test_all_selected_dense_sparse_forward_and_backward(self):
        dense = self.build(selected_complete_blocks=99)
        sparse = copy.deepcopy(dense)
        x1 = torch.randn(2, 19, 24, dtype=torch.float64, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        seg = torch.tensor([[0] * 11 + [1] * 5 + [-1] * 3, [2] * 7 + [3] * 8 + [2] * 4])
        a, _, ds = dense(x1, seg, mode='dense')
        b, _, es = sparse(x2, seg, mode='sparse')
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
        weight = torch.randn_like(a)
        (a * weight).sum().backward(); (b * weight).sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-10, atol=1e-11)
        for name, parameter in dense.named_parameters():
            if name.startswith('indexer.'):
                self.assertIsNone(parameter.grad)
                self.assertIsNone(dict(sparse.named_parameters())[name].grad)
            else:
                torch.testing.assert_close(parameter.grad, dict(sparse.named_parameters())[name].grad, rtol=1e-10, atol=1e-11)
        self.assertEqual(ds['logical_kept_pairs'], es['logical_kept_pairs'])

    def test_gqa_mapping_against_head_loop(self):
        module = self.build()
        x = torch.randn(1, 11, 24, dtype=torch.float64)
        seg = torch.zeros(1, 11, dtype=torch.long)
        actual, _, _ = module(x, seg)
        raw_q = F.linear(x[0], module.q_proj.weight).reshape(11, 4, 8)
        raw_k = F.linear(x[0], module.k_proj.weight).reshape(11, 2, 8)
        values = F.linear(x[0], module.v_proj.weight).reshape(11, 2, 8)
        norm = lambda value, gain: value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * (1 + gain)
        q = _rope(norm(raw_q, module.q_norm.weight), torch.arange(11), 4, 1e7)
        k = _rope(norm(raw_k, module.k_norm.weight), torch.arange(11), 4, 1e7)
        rows = []
        for i in range(11):
            heads = []
            for h in range(4):
                key_head = h // 2
                prob = ((k[:i + 1, key_head] * q[i, h]).sum(-1) / 8**0.5).softmax(-1)
                heads.append((prob[:, None] * values[:i + 1, key_head]).sum(0))
            rows.append(torch.stack(heads).flatten())
        expected = F.linear(torch.stack(rows) * F.linear(x[0], module.gate_proj.weight).sigmoid(), module.out_proj.weight)
        torch.testing.assert_close(actual[0], expected, rtol=1e-11, atol=1e-12)

    def test_sparse_causality_boundaries_padding_and_repeated_labels(self):
        module = self.build()
        segments = torch.tensor([[0] * 13 + [1] * 9 + [-1] * 2 + [0] * 7])
        x = torch.randn(1, 31, 24, dtype=torch.float64)
        original, _, stats = module(x, segments, mode='sparse')
        altered = x.clone(); altered[:, 7:13] += 100
        future, _, _ = module(altered, segments, mode='sparse')
        torch.testing.assert_close(original[:, :7], future[:, :7], rtol=0, atol=0)
        altered = x.clone(); altered[:, :13] -= 100
        independent, _, _ = module(altered, segments, mode='sparse')
        torch.testing.assert_close(original[:, 13:], independent[:, 13:], rtol=0, atol=0)
        self.assertEqual(stats['segments'], 3)
        self.assertTrue(bool((original[:, 22:24] == 0).all()))
        separated, _, _ = module(x[:, 24:], torch.zeros(1, 7, dtype=torch.long), mode='sparse')
        torch.testing.assert_close(original[:, 24:], separated, rtol=0, atol=0)

    def test_auxiliary_isolation_and_loss_mask_denominator(self):
        module = self.build()
        x = torch.randn(1, 21, 24, dtype=torch.float64, requires_grad=True)
        seg = torch.zeros(1, 21, dtype=torch.long)
        mask = torch.ones(1, 21, dtype=torch.bool)
        _, aux, stats = module(x, seg, mode='sparse', loss_mask=mask)
        aux.backward()
        self.assertEqual(stats['aux_loss_query_count'], 21)
        self.assertIsNone(x.grad)
        for name, p in module.named_parameters():
            if name.startswith('indexer.'):
                self.assertIsNotNone(p.grad)
                self.assertTrue(bool(torch.isfinite(p.grad).all()))
                self.assertGreater(float(p.grad.abs().sum()), 0)
            else:
                self.assertIsNone(p.grad)
        # First three query rows have no blocks but must remain in N.
        shortened_mask = mask.clone(); shortened_mask[:, :3] = False
        _, short_aux, _ = module(x, seg, mode='sparse', loss_mask=shortened_mask)
        torch.testing.assert_close(aux.detach() * 21, short_aux.detach() * 18)
        module.zero_grad(set_to_none=True)
        output, _, _ = module(x, seg, mode='sparse')
        output.square().sum().backward()
        self.assertIsNotNone(x.grad)
        for name, p in module.named_parameters():
            if name.startswith('indexer.'):
                self.assertIsNone(p.grad)
            else:
                self.assertIsNotNone(p.grad)
                self.assertTrue(bool(torch.isfinite(p.grad).all()))

    def test_empty_padding_and_single_block_zero_aux(self):
        module = self.build()
        for length in (0, 1, 3, 4, 5, 7):
            x = torch.randn(1, length, 24, dtype=torch.float64, requires_grad=True)
            seg = torch.zeros(1, length, dtype=torch.long)
            output, aux, _ = module(x, seg, mode='sparse')
            self.assertTrue(bool(torch.isfinite(output).all()))
            self.assertEqual(float(aux.detach()), 0)
        x = torch.randn(2, 9, 24, dtype=torch.float64, requires_grad=True)
        seg = torch.full((2, 9), -1, dtype=torch.long)
        output, aux, stats = module(x, seg, mode='sparse')
        (output.sum() + aux).backward()
        self.assertTrue(bool((output == 0).all()))
        self.assertTrue(bool((x.grad == 0).all()))
        self.assertEqual(stats['valid_tokens'], 0)
        self.assertEqual(stats['segments'], 0)
        self.assertEqual(float(aux.detach()), 0)

    def test_first_sparse_boundary_can_drop_current_complete_block(self):
        module = self.build(selected_complete_blocks=64)
        scores = -torch.arange(65, dtype=torch.float64).expand(260, -1)
        visible = (torch.arange(65)[None, :] * 4 + 3) <= torch.arange(260)[:, None]
        chosen, mask = module._sparse_selection(scores, visible)
        for query in (255, 256, 258):
            self.assertTrue(bool(mask[query, :query + 1].all()))
        self.assertEqual(int(mask[259].sum()), 256)
        self.assertFalse(bool(chosen[259, 64]))
        self.assertFalse(bool(mask[259, 256:260].any()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
