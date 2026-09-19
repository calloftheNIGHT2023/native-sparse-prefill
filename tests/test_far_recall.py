import sys,unittest
from pathlib import Path
import torch
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from far_recall import static_support,ancestors,family,near_control,removed_control,oracle_support,symbolic_lookup,additive_mask
from joint_attention import JointAttention

def small_config():
    return dict(sequence_length=128,layers=2,block_size=2,selected_blocks=4,records=4,
        num_keys=16,num_values=8,vocab_size=256,key_base=16,value_base=96,filler_base=128,
        record_marker=1,question_marker=2,answer_marker=3,separator=4,evidence_start=16,
        evidence_end_exclusive=48,seed=2026091405,near_offset=8)

class FarRecallTests(unittest.TestCase):
    def test_stream_seed_collision_fixed_without_rewriting_legacy(self):
        cfg=small_config(); cfg['seed']=2026091406
        a,_,_=family(cfg,'train_stream',26273); b,_,_=family(cfg,'train_stream',50281)
        self.assertTrue(torch.equal(a,b))
        cfg['family_rng']='numpy_pcg64'
        a,_,_=family(cfg,'train_stream',26273); b,_,_=family(cfg,'train_stream',50281)
        self.assertFalse(torch.equal(a,b))
        c,_,_=family(cfg,'train_stream',26273); self.assertTrue(torch.equal(a,c))

    def test_ancestors_include_indirect_paths(self):
        m=torch.eye(5,dtype=torch.bool)
        m[4,3]=True; m[3,1]=True
        r1,_=ancestors(m,4,1); r2,_=ancestors(m,4,2)
        self.assertFalse(r1[1]); self.assertTrue(r2[1])

    def test_full_sized_six_layer_isolation(self):
        cfg=small_config(); cfg.update(sequence_length=2048,layers=6,block_size=4,selected_blocks=64,
            evidence_start=64,evidence_end_exclusive=448,records=16,num_keys=64,num_values=16)
        _,_,mask=static_support(cfg); reach,_=ancestors(mask,2047,6)
        self.assertFalse(reach[64:448].any())
        self.assertTrue(reach[:4].all())

    def test_counterfactual_labels_have_identical_visible_inputs(self):
        cfg=small_config(); _,_,mask=static_support(cfg); reach,_=ancestors(mask,127,2)
        x,y,meta=family(cfg,'test',0)
        self.assertTrue(torch.equal(x[:,reach],x[:1,reach].expand(len(y),-1)))
        self.assertEqual(torch.unique(x,dim=0).shape[0],cfg['num_values'])
        self.assertTrue(torch.equal(y,symbolic_lookup(x,cfg)))
        self.assertTrue(torch.equal(removed_control(x,meta,cfg),removed_control(x,meta,cfg)[:1].expand_as(x)))
        near,p=near_control(x,meta,cfg); self.assertTrue(reach[p])
        self.assertTrue(torch.equal(symbolic_lookup(near,cfg),y))
        other,_,_=family(cfg,'validation',0); self.assertFalse(torch.equal(x,other))

    def test_key_last_format_and_oracle(self):
        cfg=small_config(); cfg.update(query_format='key_last',family_rng='numpy_pcg64')
        x,y,meta=family(cfg,'test',0)
        self.assertTrue((x[:,-1]==meta['target_key']).all())
        self.assertTrue(torch.equal(symbolic_lookup(x,cfg),y))
        near,_=near_control(x,meta,cfg); self.assertTrue(torch.equal(symbolic_lookup(near,cfg),y))

    def test_custom_mask_matches_existing_wrapper_and_blocks_gradient(self):
        torch.set_num_threads(2); torch.manual_seed(941); cfg=small_config()
        c=GPTNeoXConfig(vocab_size=256,hidden_size=32,num_hidden_layers=2,num_attention_heads=4,
            intermediate_size=64,max_position_embeddings=128,attention_dropout=0,hidden_dropout=0)
        c._attn_implementation='eager'; model=GPTNeoXForCausalLM(c).double().eval()
        layout,selected,mask=static_support(cfg); x,_,meta=family(cfg,'test',0)
        custom=additive_mask(mask,torch.float64)
        with torch.no_grad():
            outputs=model(x,attention_mask=custom,use_cache=False).logits[:,-1]
            torch.testing.assert_close(outputs,outputs[:1].expand_as(outputs),rtol=0,atol=0)
            standard=model(x[:1],attention_mask=custom,use_cache=False).logits
            wrapped_cfg=dict(cfg,index_heads=2,index_head_dim=8,rotary_dim=4,late_query_start=64,
                selection_rule='sink_recent',indexer_calibration=dict(affine=True,scaled=True))
            wrapper=JointAttention(model,wrapped_cfg); wrapper.indexers.double(); wrapper.reset('sparse',False)
            wrapped=model(x[:1],use_cache=False).logits; wrapper.restore()
            torch.testing.assert_close(wrapped,standard,rtol=1e-11,atol=1e-12)
        embeds=model.get_input_embeddings()(x[:1]).detach().requires_grad_()
        out=model(inputs_embeds=embeds,attention_mask=custom,use_cache=False).logits[0,-1,96]
        grad=torch.autograd.grad(out,embeds)[0]
        self.assertEqual(float(grad[0,meta['value_position']].abs().max()),0.)
        dense=model(x[:2],use_cache=False).logits[:,-1]
        self.assertGreater(float((dense[1]-dense[0]).abs().max()),1e-10)
        oracle=oracle_support(layout,selected,meta,cfg)
        self.assertEqual(int(oracle[-1].sum()),int(mask[-1].sum()))
        op=model(x[:2],attention_mask=additive_mask(oracle,torch.float64),use_cache=False).logits[:,-1]
        self.assertGreater(float((op[1]-op[0]).abs().max()),1e-10)

if __name__=='__main__': unittest.main()
