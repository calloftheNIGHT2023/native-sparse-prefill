import copy,math,sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from zoology_entry import configuration,LanguageModel,set_determinism
from zoology_sparse_schedule import install
from router_pressure_models import ControlRouter,fixed_mask

class PressureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);set_determinism(123);self.m=LanguageModel(configuration(128).model).double();install(self.m,'native');self.m.eval()
    def test_fixed_mask_budget_causality_and_source_pressure(self):
        mask=fixed_mask(128,'cpu');p=torch.arange(128);self.assertFalse(mask.triu(1).any());self.assertLessEqual(int(mask.sum(-1).max()),8);self.assertTrue(mask[p,p].all());self.assertTrue(mask[p[1:],p[:-1]].all())
        self.assertTrue(mask[40,[1,3,5,7,9,11]].all());self.assertEqual(int(mask[40,1:32:2].sum()),6)
    def test_dense_reference_forward_and_gradient(self):
        ref=copy.deepcopy(self.m);install(ref,'dense');ref.eval();r=ControlRouter(self.m,'dense');x=torch.randint(0,256,(2,128));a=self.m(x);b=ref(x);torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9);a.square().mean().backward();b.square().mean().backward()
        for p,q in zip(self.m.parameters(),ref.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-8)
        r.restore()
    def test_fixed_reference_forward_and_gradient(self):
        mha=self.m.backbone.layers[0].sequence_mixer;r=ControlRouter(self.m,'fixed_values6');x=torch.randn(2,128,128,dtype=torch.double,requires_grad=True);a=mha(x);q,k,v=mha.Wqkv(x).chunk(3,-1);prob=(q@k.transpose(-1,-2)/math.sqrt(128)).masked_fill(~fixed_mask(128,'cpu'),-torch.inf).softmax(-1);b=mha.out_proj(prob@v);torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
        params=[x,*mha.parameters()];ga=torch.autograd.grad(a.square().mean(),params,retain_graph=True);gb=torch.autograd.grad(b.square().mean(),params)
        for p,q in zip(ga,gb):torch.testing.assert_close(p,q,atol=1e-10,rtol=1e-8)
        r.restore()
    def test_no_future_in_fixed_model(self):
        r=ControlRouter(self.m,'fixed_values6');x=torch.randint(0,256,(2,128));y=x.clone();y[:,80:]=(y[:,80:]+1)%256;torch.testing.assert_close(self.m(x)[:,:80],self.m(y)[:,:80],atol=0,rtol=0)
        emb=self.m.backbone.embeddings(x).detach().requires_grad_(True);self.m.lm_head(self.m.backbone.layers_forward(emb))[:,79].sum().backward();self.assertEqual(float(emb.grad[:,80:].abs().max()),0);r.restore()
if __name__=='__main__':unittest.main()
