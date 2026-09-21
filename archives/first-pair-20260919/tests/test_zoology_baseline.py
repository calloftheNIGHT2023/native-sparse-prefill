import json,sys,unittest,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from zoology_entry import UPSTREAM,configuration,LanguageModel,multiquery_ar,set_determinism,prepare_data

class ZoologyTests(unittest.TestCase):
    def test_longer_configuration_changes_only_length_and_required_positions(self):
        original=configuration().model_dump(serialize_as_any=True)
        longer=configuration(128).model_dump(serialize_as_any=True)
        longer['model']['max_position_embeddings']=64
        for split in ['train_configs','test_configs']:
            for segment in longer['data'][split]:segment['input_seq_len']=64
        self.assertEqual(original,longer)
        self.assertEqual(configuration().model.max_position_embeddings,64)
        with self.assertRaises(ValueError):configuration(256)
    def test_128_positions_and_supervision(self):
        set_determinism(123);model=LanguageModel(configuration(128).model).eval()
        data=multiquery_ar(vocab_size=256,num_examples=8,input_seq_len=128,seed=42,num_kv_pairs=4)
        self.assertEqual(tuple(model(data.inputs).shape),(8,128,256))
        for x,y in zip(data.inputs,data.labels):
            mapping={int(x[i]):int(x[i+1]) for i in range(0,8,2)}
            queries=torch.where(y!=-100)[0];self.assertEqual(len(queries),4)
            for q in queries:self.assertEqual(int(y[q]),mapping[int(x[q])])
        emb=model.backbone.embeddings(data.inputs[:1]).detach().requires_grad_(True)
        model.lm_head(model.backbone.layers_forward(emb))[:,90].square().sum().backward()
        self.assertEqual(float(emb.grad[:,91:].abs().max()),0)
        self.assertGreater(float(emb.grad[:,:90].abs().max()),0)
    def test_pinned_sources_only_platform_changes(self):
        manifest=json.loads((UPSTREAM/'upstream-manifest.json').read_text())
        for record in manifest['files']:
            p=UPSTREAM/record['path']
            if record['path']=='zoology/model.py':
                original=(UPSTREAM/'upstream-original/model.py').read_bytes()
                self.assertEqual(hashlib.sha256(original).hexdigest(),record['sha256'])
                self.assertEqual(p.read_bytes(),original.replace(b"device='cuda',",b"device='cpu',"))
            elif record['path']=='zoology/data/utils.py':
                original=(UPSTREAM/'upstream-original/data-utils.py').read_bytes()
                self.assertEqual(hashlib.sha256(original).hexdigest(),record['sha256'])
                expected=original.replace(b'size=len(config.train_configs))',b'size=len(config.train_configs), dtype=np.int64)').replace(b'size=len(config.test_configs))',b'size=len(config.test_configs), dtype=np.int64)')
                self.assertEqual(p.read_bytes(),expected)
            else:self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),record['sha256'])
    def test_windows_validation_seed_range(self):
        cfg=configuration();cfg.data.train_configs[0].num_examples=16;cfg.data.test_configs[0].num_examples=8
        set_determinism(cfg.seed);train,validation=prepare_data(cfg.data)
        self.assertEqual(len(train.dataset.segments[0]),16);self.assertEqual(len(validation.dataset.segments[0]),8)
    def test_labels_at_queries_read_prefix_and_no_extra_shift(self):
        set_determinism(123)
        data=multiquery_ar(vocab_size=256,num_examples=32,input_seq_len=64,seed=12,num_kv_pairs=4)
        for x,y in zip(data.inputs,data.labels):
            mapping={int(x[i]):int(x[i+1]) for i in range(0,8,2)}
            self.assertEqual(len(mapping),4);self.assertEqual(len(set(mapping.values())),4)
            queries=torch.where(y!=-100)[0];self.assertEqual(len(queries),4)
            for q in queries:
                self.assertGreaterEqual(int(q),8);self.assertEqual(mapping[int(x[q])],int(y[q]))
                self.assertNotEqual(int(x[q]),int(y[q]))
    def test_model_ties_embeddings_and_has_causal_gradient(self):
        set_determinism(123);model=LanguageModel(configuration().model).eval()
        self.assertIs(model.lm_head.weight,model.backbone.embeddings.word_embeddings.weight)
        self.assertTrue(all(p.device.type=='cpu' for p in model.parameters()))
        emb=model.backbone.embeddings(torch.randint(0,256,(1,64))).detach().requires_grad_(True)
        hidden=model.backbone.layers_forward(emb);model.lm_head(hidden)[:,20].square().sum().backward()
        self.assertEqual(float(emb.grad[:,21:].abs().max()),0)
        self.assertGreater(float(emb.grad[:,:20].abs().max()),0)

if __name__=='__main__':unittest.main()
