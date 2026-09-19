import math,sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from router_author_control import author_configs,make_model,generated
from router_author_sparse import build,initial_indexers
from router_pressure_models import fixed_mask
from frozen_routing import select
from zoology_entry import set_determinism

class AuthorSparse(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);set_determinism(123);self.cfg=author_configs()[1];self.initial=make_model(self.cfg).state_dict();self.ix=initial_indexers()
    def test_warm_forward_matches_unmodified_upstream(self):
        a,ix,r=build(self.cfg,self.initial,self.ix,'warm4_selectedkl_r16','cpu');b=make_model(self.cfg,self.initial);a.eval();b.eval();x=generated(2026091624,2)['inputs'];torch.testing.assert_close(a(x),b(x),rtol=5e-5,atol=3e-6);sum(r.losses).backward();self.assertTrue(all(p.grad is None for p in a.parameters()));self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in ix.parameters()));r.restore()
    def test_sparse_outputs_and_gradients_match_independent_masked_reference(self):
        for method in ['warm4_selectedkl_r16','ksa_additive_r16','fixed_values6']:
            m,ix,r=build(self.cfg,self.initial,self.ix,method,'cpu');m.double().eval();ix.double();r.epoch=4;mha=m.backbone.layers[0].sequence_mixer;x=torch.randn(1,256,128,dtype=torch.double,requires_grad=True);a=mha(x);q,k,v=mha.Wqkv(x).chunk(3,-1);logits=q@k.transpose(-1,-2)/math.sqrt(128)
            if method=='fixed_values6':mask=fixed_mask(256,'cpu').unsqueeze(0);parameters=[x,*mha.parameters()]
            else:
                s=ix[0](x) if method.startswith('warm') else ix[0].query(x)@ix[0].key(x).transpose(-1,-2)/4
                mask=select(s);parameters=[x,*mha.parameters()]
                if method.startswith('ksa'):logits=logits+s;parameters+=list(ix[0].parameters())
            b=mha.out_proj(logits.masked_fill(~mask,-torch.inf).softmax(-1)@v);torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
            da=torch.autograd.grad(a.square().sum(),parameters,retain_graph=True);db=torch.autograd.grad(b.square().sum(),parameters)
            for ga,gb in zip(da,db):torch.testing.assert_close(ga,gb,rtol=1e-7,atol=1e-9)
            self.assertTrue(all(not c['dense_main'] and c['main_score_entries']==256*8 for c in r.cost));r.restore()
    def test_causal_and_fixed_budget(self):
        mask=fixed_mask(256,'cpu');self.assertFalse(mask.triu(1).any());self.assertEqual(int(mask[-1].sum()),8);self.assertEqual(int(mask[-1,1:32:2].sum()),6)
        m,ix,r=build(self.cfg,self.initial,self.ix,'ksa_additive_r16','cpu');m.eval();x=generated(2026091625,1)['inputs'];y=x.clone();y[:,100:]=(y[:,100:]+17)%8192;torch.testing.assert_close(m(x)[:,:100],m(y)[:,:100],rtol=0,atol=0)
        emb=m.backbone.embeddings(x).detach().requires_grad_(True);m.backbone.layers_forward(emb)[:,99].square().sum().backward();self.assertEqual(float(emb.grad[:,100:].abs().max()),0.);r.restore()

if __name__=='__main__':unittest.main()
