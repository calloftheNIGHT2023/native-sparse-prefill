import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sparse_reference import make_layout
from routing_rules import route_blocks

class RoutingTests(unittest.TestCase):
    def test_exact_budget_visibility_and_forced_recent(self):
        torch.manual_seed(140); layout=make_layout([0]*19+[1]*17,4)
        scores=torch.randn(36,len(layout.block_tokens))
        for rule in ['learned','learned_recent','sink_recent']:
            chosen=route_blocks(scores,layout,2,rule)
            torch.testing.assert_close(chosen.sum(-1),layout.visible_blocks.sum(-1).clamp_max(2))
            self.assertFalse(bool((chosen&~layout.visible_blocks).any()))
        chosen=route_blocks(scores,layout,2,'learned_recent')
        for i in range(36):
            ids=layout.visible_blocks[i].nonzero().flatten()
            if len(ids): self.assertTrue(bool(chosen[i,ids[-1]]))

if __name__=='__main__': unittest.main()
