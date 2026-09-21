"""Independent correctness checks for the diagnostic shortlist baseline."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from block_candidate_topk import select_block_topk, block_summaries, coarse_scores


class CandidateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(412)
        torch.set_num_threads(2)

    def test_against_independent_loop(self):
        q, k = [torch.randn(2, 37, 16) for _ in range(2)]
        for method in ['mean', 'minmax']:
            actual = select_block_topk(q, k, 8, 2, method, 11)
            for g in range(2):
                for i in range(37):
                    ranked = []
                    for b in range(i // 8):
                        keys = k[g, b * 8:(b + 1) * 8]
                        score = (keys.mean(0) * q[g, i]).sum() if method == 'mean' else torch.maximum(
                            keys.min(0).values * q[g, i], keys.max(0).values * q[g, i]).sum()
                        ranked.append((float(score), b))
                    chosen = [b for _, b in sorted(ranked, reverse=True)[:2]]
                    candidates = [j for b in chosen + [i // 8] for j in range(b * 8, (b + 1) * 8) if j <= i - 2]
                    remote = sorted(candidates, key=lambda j: float(q[g, i] @ k[g, j]), reverse=True)[:6]
                    expected = {j for j in [i, i - 1] + remote if j >= 0}
                    self.assertEqual(expected, set(actual[g, i][actual[g, i] >= 0].tolist()))

    def test_no_future_leakage(self):
        q, k = [torch.randn(2, 37, 16) for _ in range(2)]
        changed = k.clone(); changed[:, 19:] = torch.randn_like(changed[:, 19:]) * 100
        for method in ['mean', 'minmax']:
            a = select_block_topk(q, k, 8, 2, method)
            b = select_block_topk(q, changed, 8, 2, method)
            self.assertTrue(torch.equal(a[:, :19], b[:, :19]))

    def test_all_blocks_recovers_exact(self):
        q, k = [torch.randn(2, 37, 16) for _ in range(2)]
        for method in ['mean', 'minmax']:
            actual = select_block_topk(q, k, 8, 8, method)
            for g in range(2):
                for i in range(37):
                    remote = sorted(range(max(i - 1, 0)), key=lambda j: float(q[g, i] @ k[g, j]), reverse=True)[:6]
                    self.assertEqual({j for j in [i, i - 1] + remote if j >= 0},
                                     set(actual[g, i][actual[g, i] >= 0].tolist()))

    def test_bound_and_short_sequences(self):
        for n in [1, 2, 7, 16, 37]:
            q, k = [torch.randn(2, n, 16) for _ in range(2)]
            upper = coarse_scores(q, block_summaries(k, 8), 'minmax')
            for b in range(n // 8):
                true = (q @ k[:, b * 8:(b + 1) * 8].transpose(1, 2) / 4).amax(-1)
                self.assertTrue(bool((upper[:, :, b] + 1e-5 >= true).all()))
            ids = select_block_topk(q, k, 8, 2)
            for i in range(n):
                self.assertTrue(bool(((ids[:, i] <= i) & (ids[:, i] >= -1)).all()))

    def test_shifted_partition_independent_reference(self):
        q, k = [torch.randn(2, 37, 16) for _ in range(2)]
        for offset in [1, 3, 7]:
            for method in ['mean', 'minmax']:
                actual = select_block_topk(q, k, 8, 2, method, 11, offset)
                changed = k.clone(); changed[:, 19:] *= -100
                self.assertTrue(torch.equal(actual[:, :19], select_block_topk(q, changed, 8, 2, method, 11, offset)[:, :19]))
                for g in range(2):
                    for i in range(37):
                        current = (i + offset) // 8
                        blocks = {b: [j for j in range(37) if (j + offset) // 8 == b] for b in range(current + 1)}
                        ranks = []
                        for b in range(current):
                            keys = k[g, blocks[b]]
                            score = (keys.mean(0) * q[g, i]).sum() if method == 'mean' else torch.maximum(
                                keys.min(0).values * q[g, i], keys.max(0).values * q[g, i]).sum()
                            ranks.append((float(score), b))
                        chosen = [b for _, b in sorted(ranks, reverse=True)[:2]] + [current]
                        candidates = [j for b in chosen for j in blocks[b] if j <= i - 2]
                        remote = sorted(candidates, key=lambda j: float(q[g, i] @ k[g, j]), reverse=True)[:6]
                        self.assertEqual({j for j in [i, i-1] + remote if j >= 0}, set(actual[g, i][actual[g, i] >= 0].tolist()))


if __name__ == '__main__':
    unittest.main()
