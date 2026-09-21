import json,sys,unittest
from pathlib import Path
import torch
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from far_recall import family,static_support,ancestors
from recall_supervision import add_training_queries,query_loss
from run_recall_learnability import prediction

class SupervisionTests(unittest.TestCase):
    def config(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'configs/far-recall-cpu-stream-v2.json').read_text())
        c['training_query_positions']=[63,83,103,127]
        return c
    def test_answers_come_from_earlier_records_no_input_leak(self):
        c=self.config();x,y,m=family(c,'validation',0);xx,yy=add_training_queries(x,m,c)
        self.assertTrue(torch.equal(x[:,:48],xx[:,:48]))
        self.assertTrue(torch.equal(x[:,-3:],xx[:,-3:]))
        self.assertTrue(torch.equal(yy[:,-1],y+c['value_base']))
        for qi,q in enumerate(c['training_query_positions']):
            self.assertTrue((xx[:,q]<c['value_base']).all())
            for row,label in zip(xx,yy[:,qi]):
                matches=[p for p in m['record_positions'] if row[p+1]==row[q]]
                self.assertEqual(len(matches),1);self.assertLess(matches[0]+2,q)
                self.assertEqual(row[matches[0]+2],label)
        _,_,mask=static_support(c);reach,_=ancestors(mask,127,2)
        self.assertFalse(reach[m['value_position']])
    def test_last_loss_exact_and_all_queries_receive_gradients(self):
        c=self.config();x,_,m=family(c,'validation',1);xx,yy=add_training_queries(x,m,c)
        mc=GPTNeoXConfig(vocab_size=256,hidden_size=32,num_hidden_layers=2,num_attention_heads=4,intermediate_size=64,max_position_embeddings=128,attention_dropout=0,hidden_dropout=0)
        mc._attn_implementation='eager';model=GPTNeoXForCausalLM(mc).eval()
        c['supervision']='last_only'
        self.assertTrue(torch.equal(query_loss(model,xx,yy,c),torch.nn.functional.cross_entropy(prediction(model,xx),yy[:,-1])))
        c['supervision']='all_queries'; loss=query_loss(model,xx,yy,c);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(model.embed_out.weight.grad.norm()),0)

if __name__=='__main__':unittest.main()
