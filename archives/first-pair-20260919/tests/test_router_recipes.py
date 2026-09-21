import copy,math,sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from zoology_entry import configuration,LanguageModel,set_determinism
from zoology_sparse_schedule import install
from frozen_routing import TokenIndexer,select
from router_recipe_baselines import RecipeRouter

class Recipes(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);set_determinism(123);self.m=LanguageModel(configuration().model).double();install(self.m,'native');self.m.eval();self.ix=torch.nn.ModuleList([TokenIndexer(128,16).double() for _ in range(2)])
    def test_warmup_equals_dense_and_aux_is_isolated(self):
        ref=copy.deepcopy(self.m);install(ref,'dense');ref.eval();r=RecipeRouter(self.m,self.ix,'warm4_selectedkl_r16');x=torch.randint(0,256,(2,16));a=self.m(x);b=ref(x);torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
        sum(r.losses).backward();self.assertTrue(all(p.grad is None for p in self.m.parameters()));self.assertTrue(any(p.grad.abs().sum()>0 for p in self.ix.parameters()));r.restore()
    def test_selected_kl_is_isolated_and_no_dense_main_scores(self):
        r=RecipeRouter(self.m,self.ix,'warm4_selectedkl_r16');r.epoch=4;self.m(torch.randint(0,256,(2,16)));sum(r.losses).backward()
        self.assertTrue(all(p.grad is None for p in self.m.parameters()));self.assertTrue(all(c['main_score_entries']==2*16*8 and not c['dense_main'] for c in r.cost));self.assertTrue(any(p.grad.abs().sum()>0 for p in self.ix.parameters()));r.restore()
    def test_ksa_matches_dense_mask_reference_and_gradients(self):
        mha=self.m.backbone.layers[0].sequence_mixer;ix=self.ix[0];x=torch.randn(2,16,128,dtype=torch.double,requires_grad=True);r=RecipeRouter(self.m,self.ix,'ksa_additive_r16');actual=mha(x)
        q,k,v=mha.Wqkv(x).chunk(3,-1);s=ix.query(x)@ix.key(x).transpose(-1,-2)/4;mask=select(s);att=(q@k.transpose(-1,-2)/math.sqrt(128)+s).masked_fill(~mask,-torch.inf).softmax(-1);expected=mha.out_proj(att@v)
        torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-9);params=[x,*mha.parameters(),*ix.parameters()]
        a=torch.autograd.grad(actual.square().sum(),params,retain_graph=True);b=torch.autograd.grad(expected.square().sum(),params)
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,atol=1e-9,rtol=1e-7)
        self.assertGreater(float(a[-4].abs().sum()),0);self.assertFalse(r.losses);r.restore()
    def test_ksa_causal_forward_and_gradient(self):
        r=RecipeRouter(self.m,self.ix,'ksa_additive_r16');x=torch.randint(0,256,(2,16));y=x.clone();y[:,9:]=(y[:,9:]+1)%256
        torch.testing.assert_close(self.m(x)[:,:9],self.m(y)[:,:9],atol=0,rtol=0)
        emb=self.m.backbone.embeddings(x).detach().requires_grad_(True);self.m.lm_head(self.m.backbone.layers_forward(emb))[:,8].square().sum().backward();self.assertEqual(float(emb.grad[:,9:].abs().max()),0);r.restore()
if __name__=='__main__':unittest.main()
