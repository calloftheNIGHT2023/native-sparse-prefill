"""Independent semantic and gradient checks, not speed benchmarks."""
import math
from pathlib import Path
import sys
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sparse_reference import (QSAIndexer, make_layout, select_blocks, sample_outside,
                              expand_blocks, teacher_block_distribution, subset_kl, sparse_core)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)

    def test_document_boundaries_complete_blocks_and_tail(self):
        labels = [0] * 7 + [1] * 5 + [0] * 3
        layout = make_layout(labels, 4)
        self.assertEqual(layout.block_tokens.tolist(), [[0, 1, 2, 3], [7, 8, 9, 10]])
        selected = select_blocks(torch.randn(15, 2), layout.visible_blocks, 999)
        expanded = expand_blocks(selected, layout)
        expected = torch.zeros(15, 15, dtype=torch.bool)
        for start, end in [(0, 7), (7, 12), (12, 15)]:
            for i in range(start, end):
                expected[i, start:i + 1] = True
        self.assertTrue(torch.equal(expanded, expected))
        self.assertEqual(layout.visible_blocks[2].sum().item(), 0)
        self.assertEqual(layout.tail_mask[3].sum().item(), 0)

    def test_gathered_core_matches_independent_dense_attention_and_gradients(self):
        layout = make_layout([0] * 11, 4)
        selected = select_blocks(torch.randn(11, 2), layout.visible_blocks, 10)
        q, k, v = [torch.randn(11, 2, 3, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        actual = sparse_core(q, k, v, selected, layout)
        logits = torch.einsum('thd,shd->ths', q, k) / math.sqrt(3)
        dense_mask = torch.ones(11, 11, dtype=torch.bool).tril()
        expected = torch.einsum('ths,shd->thd', logits.masked_fill(~dense_mask[:, None], -torch.inf).softmax(-1), v)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)

    def test_future_and_other_document_cannot_change_past_outputs(self):
        layout = make_layout([0] * 8 + [1] * 8, 4)
        x = torch.randn(16, 16)
        idx = QSAIndexer()
        q, k, v = [torch.randn(16, 2, 8) for _ in range(3)]
        selection = select_blocks(idx(x, layout), layout.visible_blocks, 1)
        original = sparse_core(q, k, v, selection, layout)
        x2, k2, v2 = x.clone(), k.clone(), v.clone()
        x2[6:] += 100
        k2[6:] += 100
        v2[6:] += 100
        other = sparse_core(q, k2, v2, select_blocks(idx(x2, layout), layout.visible_blocks, 1), layout)
        torch.testing.assert_close(original[:6], other[:6])
        x2, k2, v2 = x.clone(), k.clone(), v.clone()
        x2[:8] += 100
        k2[:8] += 100
        v2[:8] += 100
        other = sparse_core(q, k2, v2, select_blocks(idx(x2, layout), layout.visible_blocks, 1), layout)
        torch.testing.assert_close(original[8:], other[8:])

    def test_probes_are_outside_causal_and_reproducible(self):
        layout = make_layout([0] * 24 + [1] * 13, 4)
        selected = select_blocks(torch.randn(37, 9), layout.visible_blocks, 1)
        a = sample_outside(selected, layout.visible_blocks, 3, torch.Generator().manual_seed(9))
        b = sample_outside(selected, layout.visible_blocks, 3, torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse((a & selected).any())
        self.assertFalse((a & ~layout.visible_blocks).any())
        torch.testing.assert_close(a.sum(1), (layout.visible_blocks & ~selected).sum(1).clamp_max(3))

    def test_auxiliary_gradient_updates_only_indexer(self):
        layout = make_layout([0] * 20, 4)
        hidden = torch.randn(20, 16, requires_grad=True)
        q, k = [torch.randn(20, 2, 8, requires_grad=True) for _ in range(2)]
        idx = QSAIndexer()
        scores = idx(hidden, layout)
        selected = select_blocks(scores, layout.visible_blocks, 2)
        support = selected | sample_outside(selected, layout.visible_blocks, 1, torch.Generator().manual_seed(2))
        target = teacher_block_distribution(q, k, support, layout)
        loss = subset_kl(scores, target, support)
        loss.backward()
        self.assertIsNone(hidden.grad)
        self.assertIsNone(q.grad)
        self.assertIsNone(k.grad)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in idx.parameters()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in idx.parameters()))

    def test_lm_gradient_cannot_cross_discrete_selection(self):
        layout = make_layout([0] * 12, 4)
        x = torch.randn(12, 16, requires_grad=True)
        idx = QSAIndexer()
        main = torch.nn.Linear(16, 24, bias=False)
        q, k, v = main(x).reshape(12, 3, 2, 4).unbind(1)
        selected = select_blocks(idx(x, layout), layout.visible_blocks, 1)
        sparse_core(q, k, v, selected, layout).square().mean().backward()
        self.assertGreater(main.weight.grad.abs().sum().item(), 0)
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in idx.parameters()))

    def test_teacher_multihead_max_pooling_against_scalar_oracle(self):
        layout = make_layout([0] * 13, 4)
        q, k = [torch.randn(13, 2, 4, dtype=torch.float64) for _ in range(2)]
        support = select_blocks(torch.randn(13, 3), layout.visible_blocks, 2)
        got = teacher_block_distribution(q, k, support, layout)
        for i in range(13):
            blocks = support[i].nonzero().flatten().tolist()
            tokens = sorted([j for b in blocks for j in layout.block_tokens[b].tolist()] + list(range(((i + 1) // 4) * 4, i + 1)))
            if not blocks:
                self.assertEqual(got[i].sum().item(), 0)
                continue
            p = torch.stack([torch.stack([(q[i, h] * k[j, h]).sum() / 2 for j in tokens]).softmax(0) for h in range(2)]).mean(0)
            target = torch.stack([max(p[tokens.index(j)] for j in layout.block_tokens[b].tolist()) for b in blocks])
            torch.testing.assert_close(got[i, blocks], target / target.sum())

    def test_single_block_and_empty_support_kl_have_zero_gradient(self):
        scores = torch.randn(2, 3, requires_grad=True)
        support = torch.tensor([[True, False, False], [False, False, False]])
        loss = subset_kl(scores, support.float(), support)
        loss.backward()
        self.assertAlmostEqual(loss.item(), 0)
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores))

    def test_finite_difference_of_subset_kl_and_zero_outside_gradient(self):
        support = torch.tensor([[True, True, False]])
        target = torch.tensor([[0.2, 0.8, 0.0]], dtype=torch.float64)
        x = torch.tensor([[0.3, -0.2, 5.0]], dtype=torch.float64, requires_grad=True)
        subset_kl(x, target, support).backward()
        for j in range(3):
            xp, xm = x.detach().clone(), x.detach().clone()
            xp[0, j] += 1e-6
            xm[0, j] -= 1e-6
            fd = (subset_kl(xp, target, support) - subset_kl(xm, target, support)) / 2e-6
            self.assertAlmostEqual(fd.item(), x.grad[0, j].item(), places=8)
        self.assertEqual(x.grad[0, 2].item(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
