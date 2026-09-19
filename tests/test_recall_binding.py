import sys,unittest
from pathlib import Path
import torch
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from recall_binding import install_binding_control

class BindingTests(unittest.TestCase):
    def test_first_layer_only_predecessor_no_future_and_no_parameter_change(self):
        c=GPTNeoXConfig(vocab_size=128,hidden_size=32,num_hidden_layers=2,num_attention_heads=4,intermediate_size=64,max_position_embeddings=16)
        c._attn_implementation='eager';m=GPTNeoXForCausalLM(c).eval();keys=set(m.state_dict())
        install_binding_control(m,{'binding_control':'predecessor_first_layer','sequence_length':16})
        self.assertEqual(keys,set(m.state_dict()))
        x=torch.randint(0,128,(2,16))
        with torch.no_grad():
            r=m.gpt_neox(x,output_attentions=True,use_cache=False)
        attn=r.attentions[0];indices=torch.arange(16);previous=(indices-1).clamp_min(0)
        self.assertTrue(torch.equal(attn[:,:,indices,previous],torch.ones((2,4,16))))
        self.assertEqual(float(attn.sum()),128.)

if __name__=='__main__':unittest.main()
