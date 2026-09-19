import math
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sparse_reference import QSAIndexer, make_layout, select_blocks, subset_kl
from indexer_calibration import CalibratedIndexer


class CalibrationTests(unittest.TestCase):
    def test_initial_identity_scale_routing_and_gradient_isolation(self):
        torch.manual_seed(71)
        hidden = torch.randn(32, 16, requires_grad=True)
        layout = make_layout([0]*32, 4)
        torch.manual_seed(19)
        reference = QSAIndexer(16, 2, 8, 4)
        ref_scores = reference(hidden, layout)
        for affine in (False, True):
            for scaled in (False, True):
                torch.manual_seed(19)
                indexer = CalibratedIndexer(16, 2, 8, 4, affine, scaled)
                scale = 8**-0.5 if scaled else 1.
                scores = indexer(hidden, layout)
                torch.testing.assert_close(scores, ref_scores * scale, rtol=0, atol=0)
                self.assertTrue(torch.equal(select_blocks(scores, layout.visible_blocks, 3),
                                            select_blocks(ref_scores, layout.visible_blocks, 3)))
                target = layout.visible_blocks.float()
                target /= target.sum(-1, keepdim=True).clamp_min(1)
                subset_kl(scores, target, layout.visible_blocks).backward()
                self.assertIsNone(hidden.grad)
                if affine:
                    self.assertGreater(indexer.query_gain.grad.abs().sum().item(), 0)
                    self.assertGreater(indexer.key_gain.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
