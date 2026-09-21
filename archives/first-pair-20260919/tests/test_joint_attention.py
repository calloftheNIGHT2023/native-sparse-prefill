import copy, sys, unittest
from pathlib import Path
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from joint_attention import JointAttention

class JointTests(unittest.TestCase):
    def setup_models(self):
        torch.manual_seed(1400); torch.set_num_threads(2)
        c=GPTNeoXConfig(vocab_size=67,hidden_size=32,num_hidden_layers=2,num_attention_heads=4,
            intermediate_size=64,max_position_embeddings=32,attention_dropout=0,hidden_dropout=0)
        c._attn_implementation='eager'
        original=GPTNeoXForCausalLM(c).double(); model=copy.deepcopy(original)
        cfg={'sequence_length':16,'block_size':2,'selected_blocks':2,'late_query_start':8,
            'index_heads':2,'index_head_dim':8,'rotary_dim':4,'indexer_calibration':{'affine':True,'scaled':True}}
        wrapper=JointAttention(model,cfg); wrapper.indexers.double()
        x=torch.randint(0,67,(1,16))
        return original,model,wrapper,x

    def test_dense_forward_and_backward_match(self):
        original,model,wrapper,x=self.setup_models(); wrapper.reset('dense',False)
        a=original(x,use_cache=False).logits; b=model(x,use_cache=False).logits
        torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-12)
        a.square().sum().backward(); b.square().sum().backward()
        for pa,pb in zip(original.parameters(),model.parameters()):
            torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-10,atol=1e-11)

    def test_gradient_isolation_and_sparse_causality(self):
        _,model,wrapper,x=self.setup_models(); wrapper.reset('sparse',True)
        pred=model(x,use_cache=False).logits
        aux=sum(wrapper.losses); aux.backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in wrapper.indexers.parameters() if p.grad is not None),0)
        wrapper.indexers.zero_grad(set_to_none=True); wrapper.reset('sparse',False)
        pred=model(x,use_cache=False).logits; pred.square().sum().backward()
        self.assertTrue(all(p.grad is None for p in wrapper.indexers.parameters()))
        for layer in model.gpt_neox.layers:
            self.assertGreater(float(layer.attention.query_key_value.weight.grad.abs().sum()),0)
        perturbed=x.clone(); perturbed[:,10:]=(perturbed[:,10:]+1)%67
        wrapper.reset('sparse',False); other=model(perturbed,use_cache=False).logits
        torch.testing.assert_close(pred[:,:10],other[:,:10],rtol=1e-11,atol=1e-12)

    def test_gathered_joint_loss_and_gradient_match(self):
        _,model,wrapper,x=self.setup_models()
        wrapper.reset('sparse',True)
        pred=model(x,use_cache=False).logits
        loss=pred.square().sum()+sum(wrapper.losses)
        loss.backward()
        parameters=list(model.parameters())+list(wrapper.indexers.parameters())
        gradients=[p.grad.detach().clone() for p in parameters]
        for p in parameters: p.grad=None
        wrapper.cfg['core_backend']='gathered'; wrapper.reset('sparse',True)
        other=model(x,use_cache=False).logits
        other_loss=other.square().sum()+sum(wrapper.losses)
        other_loss.backward()
        # float32 softmax summation order differs after KV gather permutation.
        torch.testing.assert_close(loss,other_loss,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(pred,other,rtol=1e-5,atol=1e-7)
        for a,p in zip(gradients,parameters): torch.testing.assert_close(a,p.grad,rtol=1e-4,atol=2e-6)

    def test_self_only_removes_previous_context(self):
        _,model,wrapper,x=self.setup_models(); wrapper.reset('self_only',False)
        output=model(x,use_cache=False).logits
        perturbed=x.clone(); perturbed[:,:9]=(perturbed[:,:9]+1)%67
        wrapper.reset('self_only',False); changed=model(perturbed,use_cache=False).logits
        torch.testing.assert_close(output[:,9:],changed[:,9:],rtol=1e-11,atol=1e-12)

if __name__=='__main__': unittest.main()
