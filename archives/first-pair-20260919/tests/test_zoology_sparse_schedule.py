import sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from zoology_entry import configuration,LanguageModel,set_determinism
from zoology.mixers.attention import SelfAttention
from zoology_sparse_schedule import ScheduledAttention,SCHEDULES,install,retained_edges

class SparseScheduleTests(unittest.TestCase):
    def test_dense_matches_upstream_outputs_and_gradients(self):
        torch.manual_seed(9)
        x=torch.randn(2,16,3,1,8,dtype=torch.float64,requires_grad=True)
        y=x.detach().clone().requires_grad_(True)
        old=SelfAttention(0).eval();new=ScheduledAttention(0,'dense',dropout=0).eval()
        a=old(x);b=new(y)
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        a.square().sum().backward();b.square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=1e-11,atol=1e-11)
    def test_causal_exact_budget_local_edges_and_topk(self):
        torch.manual_seed(2);scores=torch.randn(2,1,64,64)
        layer=ScheduledAttention(0,'native');mask=layer.selection(scores)
        p=torch.arange(64);allowed=p[:,None]>=p[None,:]
        self.assertFalse(bool((mask&~allowed).any()))
        self.assertTrue(torch.equal(mask.sum(-1),torch.minimum(p+1,torch.tensor(8)).expand(2,1,64)))
        self.assertTrue(bool(mask[:,:,p,p].all()));self.assertTrue(bool(mask[:,:,p[1:],p[:-1]].all()))
        expected=torch.topk(scores[:,:,63,:62],6,-1).indices
        self.assertTrue(bool(mask[:,:,63,:].gather(-1,expected).all()))
    def test_dropped_key_and_future_gradients_are_zero(self):
        torch.manual_seed(13);x=torch.randn(1,32,3,1,8,requires_grad=True)
        layer=ScheduledAttention(0,'native',dropout=0).eval();out=layer(x);out[:,20].square().sum().backward()
        excluded=~layer.last_mask[0,0,20]
        self.assertEqual(float(x.grad[0,excluded,2].abs().max()),0.)
        self.assertEqual(float(x.grad[0,21:].abs().max()),0.)
    def test_parameter_initialization_unmodified(self):
        set_determinism(123);m=LanguageModel(configuration().model)
        before={k:v.clone() for k,v in m.state_dict().items()};install(m,'native')
        self.assertEqual(set(before),set(m.state_dict()))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in m.state_dict().items()))
    def test_matched_extra_retained_edges_and_common_final_architecture(self):
        extra={}
        for schedule in ['all_warm4','first_warm8','second_warm8']:
            extra[schedule]=sum(SCHEDULES[schedule])*(retained_edges(64,64)-retained_edges(64,8))
            for i in range(2):
                layer=ScheduledAttention(i,schedule);layer.epoch=8;self.assertFalse(layer.dense())
        self.assertEqual(len(set(extra.values())),1)

if __name__=='__main__':unittest.main()
