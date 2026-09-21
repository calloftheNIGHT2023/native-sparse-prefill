import sys
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chunked_topk_attention import ChunkedTopKAttention, select_causal_topk, selected_attention
from zoology_sparse_schedule import ScheduledAttention


class ChunkedTopKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_output_and_gradients_against_historical_dense_mask(self):
        for n, heads, budget, local, chunk in [(1,1,8,2,3), (5,2,8,2,3), (17,2,8,2,4),
                                                (19,3,3,1,5), (11,2,2,2,4), (9,1,1,1,2)]:
            with self.subTest(n=n, heads=heads, budget=budget, local=local):
                torch.manual_seed(920 + n)
                x = torch.randn(2,n,3,heads,7,dtype=torch.float64,requires_grad=True)
                y = x.detach().clone().requires_grad_(True)
                old = ScheduledAttention(0,'native',budget,local,0).eval()
                new = ChunkedTopKAttention(budget,local,0,chunk).eval()
                a,b = old(x),new(y)
                torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-12)
                upstream = torch.randn_like(a)
                da, = torch.autograd.grad(a,x,upstream)
                db, = torch.autograd.grad(b,y,upstream)
                torch.testing.assert_close(da,db,rtol=1e-10,atol=2e-12)

    def test_selection_causality_uniqueness_budget_and_dense_reference(self):
        torch.manual_seed(930)
        q,k = [torch.randn(3,37,9,dtype=torch.float64) for _ in range(2)]
        for chunk in [1,7,64]:
            ids = select_causal_topk(q,k,8,2,chunk)
            scores = torch.bmm(q,k.transpose(1,2)/9**.5)
            expected = ScheduledAttention(0,'native',8,2,0).selection(scores)
            reconstructed = torch.zeros_like(expected)
            for g in range(3):
                for i in range(37):
                    entries = ids[g,i][ids[g,i]>=0]
                    self.assertEqual(len(entries), min(i+1,8))
                    self.assertEqual(len(entries.unique()),len(entries))
                    self.assertTrue(bool((entries<=i).all()))
                    self.assertTrue(i in entries and (i==0 or i-1 in entries))
                    reconstructed[g,i,entries] = True
            self.assertTrue(torch.equal(reconstructed,expected))

    def test_first_order_finite_difference(self):
        torch.manual_seed(931)
        q,k,v = [torch.randn(1,5,3,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        ids = select_causal_topk(q,k,3,1,2)
        self.assertTrue(torch.autograd.gradcheck(lambda q,k,v: selected_attention(q,k,v,ids,query_chunk=2),
                                                (q,k,v),eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_dropout_backward_against_independent_dense_autograd(self):
        torch.manual_seed(932)
        q,k,v = [torch.randn(2,13,5,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        ids = select_causal_topk(q,k,5,2,4)
        keep = torch.rand(ids.shape) >= .3
        sparse = selected_attention(q,k,v,ids,.3,4,keep)
        # Build a dense mask and align the exact same random keep bits to its edges.
        mask = torch.zeros(2,13,13,dtype=torch.bool)
        keep_dense = torch.zeros(2,13,13,dtype=torch.float64)
        for g in range(2):
            for i in range(13):
                valid=ids[g,i]>=0
                mask[g,i,ids[g,i,valid]]=True
                keep_dense[g,i,ids[g,i,valid]]=keep[g,i,valid].double()/.7
        scores = q @ (k/5**.5).transpose(-1,-2)
        dense = (scores.masked_fill(~mask,-torch.inf).softmax(-1)*keep_dense) @ v
        torch.testing.assert_close(dense,sparse,rtol=1e-11,atol=1e-12)
        go=torch.randn_like(dense)
        expected=torch.autograd.grad(dense,(q,k,v),go,retain_graph=True)
        actual=torch.autograd.grad(sparse,(q,k,v),go)
        for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=1e-10,atol=2e-12)

    def test_no_future_or_unselected_value_gradient_and_no_future_influence(self):
        torch.manual_seed(933)
        q,k,v=[torch.randn(1,17,6,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        ids=select_causal_topk(q,k,4,2,5)
        out=selected_attention(q,k,v,ids,query_chunk=5)
        dv,=torch.autograd.grad(out[:,10].sum(),v)
        selected=set(ids[0,10].tolist())
        for j in range(17):
            if j not in selected:self.assertEqual(float(dv[0,j].abs().max()),0.0)
        k2=k.detach().clone();v2=v.detach().clone();k2[:,11:]*=100;v2[:,11:]+=1000
        ids2=select_causal_topk(q.detach(),k2,4,2,5)
        changed=selected_attention(q.detach(),k2,v2,ids2,query_chunk=5)
        torch.testing.assert_close(changed[:,:11],out.detach()[:,:11],rtol=0,atol=0)

    def test_backward_saves_compact_indices_not_dense_attention(self):
        torch.manual_seed(934)
        q,k,v=[torch.randn(2,67,11,requires_grad=True) for _ in range(3)]
        shapes=[]
        def pack(t):shapes.append(tuple(t.shape));return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            ids=select_causal_topk(q,k,8,2,16)
            out=selected_attention(q,k,v,ids,query_chunk=16)
        self.assertEqual(shapes,[(2,67,11)]*3+[(2,67,8),(0,)])
        out.sum().backward()

    def test_dropout_rng_replayed_in_backward_and_eval_disables_dropout(self):
        torch.manual_seed(935)
        x=torch.randn(1,17,3,2,5,requires_grad=True)
        layer=ChunkedTopKAttention(dropout=.3,query_chunk=4).train()
        a=layer(x);rng=torch.get_rng_state().clone();a.sum().backward()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        layer.eval();a=layer(x.detach());b=layer(x.detach())
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_invalid_configuration(self):
        for kwargs in [dict(k=1,local=2),dict(dropout=1),dict(query_chunk=0)]:
            with self.assertRaises(ValueError):ChunkedTopKAttention(**kwargs)


if __name__=='__main__':unittest.main()
