import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import GPTNeoXConfig
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_cloud_control import main,sequence_order
from run_joint_pilot import save,sha
from prepare_cloud_articles import articles

class CloudTests(unittest.TestCase):
    def test_article_boundaries(self):
        rows=['',' = Article One = ','paragraph',' = = Section = = ','body',' = Article Two = ','next']
        self.assertEqual([(a,b) for a,b,_ in articles(rows)],[(1,5),(5,7)])

    def test_shuffle_epochs(self):
        order=sequence_order(4,9,17)
        self.assertEqual(order,sequence_order(4,9,17))
        self.assertEqual(sorted(order[:4]),list(range(4)))
        self.assertEqual(sorted(order[4:8]),list(range(4)))

    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/'model'; source.mkdir(); data=root/'data'; data.mkdir()
            c=GPTNeoXConfig(vocab_size=67,hidden_size=32,num_hidden_layers=2,num_attention_heads=4,intermediate_size=64,max_position_embeddings=32,attention_dropout=0,hidden_dropout=0)
            c.save_pretrained(source)
            cfg=dict(pretrained_source=str(source),initialization='random',seed=41,sequence_length=16,
                block_size=2,selected_blocks=2,late_query_start=8,index_heads=2,index_head_dim=8,rotary_dim=4,
                indexer_calibration=dict(affine=True,scaled=True),core_backend='gathered',updates=3,
                lm_learning_rate=1e-4,weight_decay=.01,lr_warmup_updates=1,evaluation_every=2,
                context_gate_nats=.02,scope='Technical synthetic resume verification')
            config=root/'config.json'; save(config,cfg)
            g=torch.Generator().manual_seed(800)
            torch.save({k:torch.randint(0,67,(n,17),generator=g) for k,n in dict(train=4,validation=2,test=2).items()},data/'tokens.pt')
            save(data/'manifest.json',dict(sequence_length=16,tokens_sha256=sha(data/'tokens.pt')))
            def run(folder,resume=False,limit=60):
                main(SimpleNamespace(config=config,data=data,output=root/folder,device='cpu',resume=resume,max_seconds=limit))
            run('direct'); run('resumed',limit=0); run('resumed',resume=True)
            a=json.loads((root/'direct/result.json').read_text()); b=json.loads((root/'resumed/result.json').read_text())
            self.assertEqual(a['final_hash'],b['final_hash']); self.assertEqual(a['test'],b['test'])
if __name__=='__main__': unittest.main()
