from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sparse_reference import make_layout,select_blocks
from head_mixture import corrected_target,dense_restricted_target,exact_retained_mass,sampled_retained_mass


class MixtureTests(unittest.TestCase):
 def test_exact_and_census_correction(self):
  torch.manual_seed(140)
  layout=make_layout([0]*17+[1]*19,4)
  logits=torch.randn(3,36,36,dtype=torch.float64)
  support=select_blocks(torch.randn(36,len(layout.block_tokens)),layout.visible_blocks,2)
  mass=exact_retained_mass(logits,support,layout)
  target=corrected_target(logits,support,layout,mass)
  torch.testing.assert_close(target,dense_restricted_target(logits,support,layout),rtol=1e-12,atol=1e-12)
  census=sampled_retained_mass(logits,support,layout,99,torch.Generator().manual_seed(2))
  torch.testing.assert_close(census,mass,rtol=1e-12,atol=1e-12)
  self.assertTrue(torch.isfinite(target).all())
  self.assertTrue((target[~support]==0).all())

 def test_two_head_rank_reversal(self):
  # Within selected A,B the first head carries 99x more dense mass.
  dense=torch.tensor([[.80,.19,.01],[.001,.009,.99]],dtype=torch.float64)
  sparse=dense[:,:2]/dense[:,:2].sum(-1,keepdim=True)
  self.assertEqual(sparse.mean(0).argmax().item(),1)
  self.assertEqual(dense[:,:2].mean(0).argmax().item(),0)
  corrected=(sparse*dense[:,:2].sum(-1,keepdim=True)).mean(0)
  torch.testing.assert_close(corrected,dense[:,:2].mean(0))


if __name__=='__main__': unittest.main()
