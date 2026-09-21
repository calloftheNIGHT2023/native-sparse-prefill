from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sparse_reference import make_layout, select_blocks, teacher_block_distribution, sparse_core
from trace_math import prepare_trace, target_from_logits, output_error


class TraceTests(unittest.TestCase):
    def test_cached_scores_agree_with_reference_teacher_and_value_output(self):
        torch.manual_seed(44)
        layout = make_layout([0] * 13 + [1] * 8, 4)
        q, k, v = [torch.randn(21, 3, 4, dtype=torch.float64) for _ in range(3)]
        support = select_blocks(torch.randn(21, 5), layout.visible_blocks, 1)
        prepared = prepare_trace({'q': q, 'k': k, 'v': v}, layout)
        torch.testing.assert_close(target_from_logits(prepared['logits'], support, layout),
                                   teacher_block_distribution(q, k, support, layout), rtol=1e-10, atol=1e-10)
        got = sparse_core(q, k, v, support, layout).permute(1, 0, 2)
        dense = sparse_core(q, k, v, layout.visible_blocks, layout).permute(1, 0, 2)
        expected = (got[:, 5:] - dense[:, 5:]).square().sum() / dense[:, 5:].square().sum()
        self.assertAlmostEqual(output_error(prepared, support, layout, 5), expected.item(), places=10)
        self.assertLess(output_error(prepared, layout.visible_blocks, layout, 5), 1e-20)


if __name__ == '__main__':
    unittest.main(verbosity=2)
