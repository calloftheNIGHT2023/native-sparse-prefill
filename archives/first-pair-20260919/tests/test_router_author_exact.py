import copy,math,sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from router_author_control import author_configs,make_model,generated
from zoology_sparse_schedule import install
from frozen_routing import select
from zoology_entry import set_determinism

class ExactNative(unittest.TestCase):
    def test_author_exact_mask_forward_gradient_and_causality(self):
        torch.set_num_threads(1);set_determinism(123);m=make_model(author_configs()[1]);install(m,'native');m.double().eval();mha=m.backbone.layers[0].sequence_mixer;x=torch.randn(1,256,128,dtype=torch.double,requires_grad=True);a=mha(x);q,k,v=mha.Wqkv(x).chunk(3,-1);s=q@(k/math.sqrt(128)).transpose(-1,-2);mask=select(s);b=mha.out_proj(s.masked_fill(~mask,-torch.inf).softmax(-1)@v);torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10);params=[x,*mha.parameters()];da=torch.autograd.grad(a.square().sum(),params,retain_graph=True);db=torch.autograd.grad(b.square().sum(),params)
        for ga,gb in zip(da,db):torch.testing.assert_close(ga,gb,rtol=1e-7,atol=1e-9)
        self.assertFalse(mask.triu(1).any());self.assertLessEqual(int(mask.sum(-1).max()),8)
        m.float();data=generated(2026091640,1);xx=data['inputs'];yy=xx.clone();yy[:,100:]=(yy[:,100:]+11)%8192;torch.testing.assert_close(m(xx)[:,:100],m(yy)[:,:100],rtol=0,atol=0);emb=m.backbone.embeddings(xx).detach().requires_grad_(True);m.backbone.layers_forward(emb)[:,99].square().sum().backward();self.assertEqual(float(emb.grad[:,100:].abs().max()),0.)

if __name__=='__main__':unittest.main()
