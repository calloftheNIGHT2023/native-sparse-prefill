import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sparse_reference import make_layout,select_blocks,expand_blocks
from trace_math import target_from_logits
from gathered_core import gathered_attention,gather_plan


class GatherTests(unittest.TestCase):
    def test_output_gradients_teacher_and_padding(self):
        torch.manual_seed(1450); torch.set_num_threads(2)
        layout=make_layout([0]*19+[1]*17,4); n=36
        support=select_blocks(torch.randn(n,len(layout.block_tokens)),layout.visible_blocks,2)
        q,k,v=[torch.randn(3,n,8,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        logits=q@k.transpose(-1,-2)/8**.5
        mask=expand_blocks(support,layout)
        dense=logits.masked_fill(~mask[None],-torch.inf).softmax(-1)@v
        sparse,target,stats=gathered_attention(q,k,v,support,layout,softmax_fp32=False)
        torch.testing.assert_close(dense,sparse,rtol=1e-12,atol=1e-12)
        grad=torch.randn_like(dense)
        dg=torch.autograd.grad(dense,(q,k,v),grad,retain_graph=True)
        sg=torch.autograd.grad(sparse,(q,k,v),grad)
        for a,b in zip(dg,sg): torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-12)
        torch.testing.assert_close(target,target_from_logits(logits.detach(),support,layout),rtol=1e-12,atol=1e-12)
        self.assertFalse(target.requires_grad)
        self.assertEqual(stats['computed_main_qk_pairs'],int(mask.sum()))
        ids,valid,_,_=gather_plan(support,layout)
        reconstructed=torch.zeros_like(mask)
        rows=torch.arange(n)[:,None].expand_as(ids)
        reconstructed[rows[valid],ids[valid]]=True
        self.assertTrue(torch.equal(mask,reconstructed))
        self.assertGreater(stats['padded_gather_slots'],0)

    def test_tail_only_short_documents(self):
        layout=make_layout([0]*2+[1]*3,4); support=layout.visible_blocks
        q,k,v=[torch.randn(2,5,8,dtype=torch.float64) for _ in range(3)]
        out,target,_=gathered_attention(q,k,v,support,layout,softmax_fp32=False)
        expected=(q@k.transpose(-1,-2)/8**.5).masked_fill(~layout.causal_mask[None],-torch.inf).softmax(-1)@v
        torch.testing.assert_close(out,expected,rtol=1e-12,atol=1e-12)
        self.assertEqual(target.shape,(5,0))

if __name__=='__main__': unittest.main()
