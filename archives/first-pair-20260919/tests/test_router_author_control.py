import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from router_author_control import author_configs,make_model,selected_logits,equivalence,generated,validate,interventions

class AuthorControlTests(unittest.TestCase):
    def test_author_configuration_retains_subclass_fields(self):
        cs=author_configs();self.assertEqual(len(cs),2)
        for c in cs:
            self.assertEqual(c['data']['train_configs'][0]['num_kv_pairs'],16);self.assertFalse(c['data']['train_configs'][0]['random_non_queries']);self.assertEqual(c['data']['batch_size'],256);self.assertEqual(c['max_epochs'],64);self.assertEqual(c['model']['state_mixer']['name'],'torch.nn.Identity')
    def test_supervised_head_matches_full_loss_and_all_gradients(self):
        torch.set_num_threads(4);equivalence('cpu')
    def test_future_tokens_have_no_effect_or_gradient(self):
        torch.set_num_threads(4);m=make_model(author_configs()[0]);m.eval();data=generated(2026091611,1);x=data['inputs'];y=data['labels'];q=int(torch.where(y[0]!=-100)[0][0]);mask=torch.full_like(y,-100);mask[0,q]=y[0,q]
        hidden=[]
        def hook(mod,args,output):output.retain_grad();hidden.append(output)
        handle=m.backbone.embeddings.register_forward_hook(hook);logits,_=selected_logits(m,x,mask);logits.square().mean().backward();self.assertEqual(float(hidden[0].grad[:,q+1:].abs().max()),0.);handle.remove()
        changed=x.clone();changed[:,q+1:]=(changed[:,q+1:]+51)%8192
        with torch.no_grad():actual,_=selected_logits(m,changed,mask)
        torch.testing.assert_close(logits,actual,rtol=0,atol=0)
    def test_interventions_preserve_questions(self):
        d=generated(2026091601,32);validate(d);swap,noisy,pairs=interventions(d)
        for i,(q,s,t) in enumerate(pairs):
            self.assertEqual(int(swap['labels'][i,q]),int(d['inputs'][i,t]));self.assertNotEqual(int(swap['labels'][i,q]),int(d['labels'][i,q]));self.assertEqual(int((swap['inputs'][i]!=d['inputs'][i]).sum()),2)

if __name__=='__main__':unittest.main()
