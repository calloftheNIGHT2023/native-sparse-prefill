import copy,sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from zoology_entry import configuration,LanguageModel,set_determinism
from zoology_sparse_schedule import install
from frozen_routing import TokenIndexer,FrozenRouter
from joint_token_routing import JointTokenRouter

class JointTokenTests(unittest.TestCase):
    def setup_model(self):
        set_determinism(123);m=LanguageModel(configuration().model).double();install(m,'native');m.eval()
        ix=torch.nn.ModuleList([TokenIndexer(128,16).double() for _ in range(2)])
        return m,ix
    def test_main_and_aux_gradient_isolation(self):
        m,ix=self.setup_model();r=JointTokenRouter(m,ix);x=torch.randint(0,256,(2,64));logits=m(x)
        sum(r.losses).backward();self.assertTrue(all(p.grad is None for p in m.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in ix.parameters()))
        ix.zero_grad();r.reset();m(x).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in ix.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters()));r.restore()
    def test_exact_gather_matches_reference_forward_and_backward(self):
        m,ix=self.setup_model();reference=copy.deepcopy(m);r=JointTokenRouter(m,ix,'exact');x=torch.randint(0,256,(2,64))
        a=m(x);b=reference(x);torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
        a.square().mean().backward();b.square().mean().backward()
        for p,q in zip(m.parameters(),reference.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-8,atol=1e-10)
        r.restore()
    def test_learned_eval_matches_frozen_wrapper(self):
        m,ix=self.setup_model();other=copy.deepcopy(m);a=JointTokenRouter(m,ix);b=FrozenRouter(other,ix)
        x=torch.randint(0,256,(2,64));torch.testing.assert_close(m(x),other(x),rtol=0,atol=0);a.restore();b.restore()
    def test_causal_forward_and_gradient(self):
        m,ix=self.setup_model();r=JointTokenRouter(m,ix);x=torch.randint(0,256,(1,64));y=x.clone();y[:,31:]=(y[:,31:]+1)%256
        torch.testing.assert_close(m(x)[:,:31],m(y)[:,:31],rtol=0,atol=0)
        emb=m.backbone.embeddings(x).detach().requires_grad_(True);m.lm_head(m.backbone.layers_forward(emb))[:,30].square().sum().backward()
        self.assertEqual(float(emb.grad[:,31:].abs().max()),0.);r.restore()
if __name__=='__main__':unittest.main()
