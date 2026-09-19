import copy,math,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from frozen_routing import TokenIndexer,FrozenRouter,select,routed_core,full_teacher,kl_loss
from zoology_entry import configuration,LanguageModel,set_determinism
from zoology_sparse_schedule import install

class FrozenRoutingTests(unittest.TestCase):
    def test_gather_forward_backward_matches_mask(self):
        torch.manual_seed(13);a=[torch.randn(2,19,16,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        b=[x.detach().clone().requires_grad_(True) for x in a];mask=select(torch.randn(2,19,19))
        one=routed_core(*a,mask);q,k,v=b;two=(q@k.transpose(-1,-2)/4).masked_fill(~mask,-torch.inf).softmax(-1)@v
        torch.testing.assert_close(one,two,rtol=1e-12,atol=1e-12);one.square().sum().backward();two.square().sum().backward()
        for x,y in zip(a,b):torch.testing.assert_close(x.grad,y.grad,rtol=1e-10,atol=1e-11)
    def test_full_rank_copy_model_equivalence(self):
        set_determinism(123);model=LanguageModel(configuration().model).double();install(model,'native');model.eval()
        copied=copy.deepcopy(model);indexers=torch.nn.ModuleList([TokenIndexer(128,128).double() for _ in range(2)])
        for layer,indexer in zip(copied.backbone.layers,indexers):indexer.initialize(layer.sequence_mixer,'copy')
        routing=FrozenRouter(copied,indexers);x=torch.randint(0,256,(2,64));torch.testing.assert_close(model(x),copied(x),rtol=1e-9,atol=1e-10);routing.restore()
    def test_auxiliary_gradient_stops_at_inputs_and_teacher(self):
        x=torch.randn(2,16,12,requires_grad=True);target=torch.randn(2,16,16,requires_grad=True).softmax(-1);target.retain_grad()
        router=TokenIndexer(12,4);loss=kl_loss(router(x),target);loss.backward()
        self.assertIsNone(x.grad);self.assertIsNone(target.grad);self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in router.parameters()))
    def test_future_tokens_do_not_change_output_or_gradient(self):
        torch.manual_seed(19);model=LanguageModel(configuration().model);install(model,'native');model.eval()
        router=FrozenRouter(model,torch.nn.ModuleList([TokenIndexer(128,16) for _ in range(2)]));x=torch.randint(0,256,(1,64));y=x.clone();y[:,31:]=(y[:,31:]+1)%256
        torch.testing.assert_close(model(x)[:,:31],model(y)[:,:31],rtol=0,atol=0)
        emb=model.backbone.embeddings(x).detach().requires_grad_(True);model.lm_head(model.backbone.layers_forward(emb))[:,30].square().sum().backward()
        self.assertEqual(float(emb.grad[:,31:].abs().max()),0.);router.restore()
    def test_svd_has_best_rank_score_matrix_residual(self):
        model=LanguageModel(configuration().model).double();mha=model.backbone.layers[0].sequence_mixer;idx=TokenIndexer(128,16).double();idx.initialize(mha,'svd')
        w=mha.Wqkv.weight;b=mha.Wqkv.bias;q=torch.cat([w[:128].T,b[:128][None]],0);k=torch.cat([w[128:256].T,b[128:256][None]],0);matrix=q@k.T/math.sqrt(128)
        iq=torch.cat([idx.query.weight.T,idx.query.bias[None]],0);ik=torch.cat([idx.key.weight.T,idx.key.bias[None]],0)
        error=(matrix-iq@ik.T/4).square().sum();expected=torch.linalg.svdvals(matrix)[16:].square().sum()
        torch.testing.assert_close(error,expected,rtol=1e-10,atol=1e-12)
if __name__=='__main__':unittest.main()
