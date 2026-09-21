"""Prepare raw train/validation corpora and fixed bounded protocol locally."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'data/flashmoba-pool-training-v0'
out.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[]
for split in ['train','validation']:
    p=ROOT/f'data/realtext-v0-r1/assets/{split}.parquet'
    dest=out/f'{split}.txt'
    dest.write_text('\n\n'.join(pq.read_table(p,columns=['text'])['text'].to_pylist()),encoding='utf-8',newline='\n')
    sources.append(dict(split=split,parquet_path=str(p.relative_to(ROOT)),parquet_sha256=sha(p),text_sha256=sha(dest)))
cfg=dict(seed=2026091545,length=2048,steps=8,validation_windows=4,block_size=128,topk=4,
    model_path='data/flashmoba-qwen-precision-v0/model',model_revision='060db6499f32faf8b98477b0a26969ef7d8b9987',
    dataset_id='Salesforce/wikitext',dataset_config='wikitext-2-raw-v1',dataset_revision='b08601e04326c79dfdd32d625aee71d232d685c3',
    model_dtype='float32',attention_dtype='bfloat16',lora_rank=8,lora_alpha=16,
    learning_rate=0.0001,optimizer='AdamW',betas=[0.9,0.999],eps=1e-8,weight_decay=0.0,
    gradient_clip=None,dropout=0.0,gradient_checkpointing=False,deterministic_backward=True,
    conditions=[dict(name=n,mode=m,bn=b) for n,m,b in [
        ('dense32','dense',32),('official32a','official',32),('fp3232','fp32',32),
        ('official64','official',64),('fp3264','fp32',64),('dense128','dense',128),
        ('official128','official',128),('fp32128','fp32',128),('official32b','official',32)]],
    scope='Bounded paired LoRA trajectory diagnostic, not independent seeds or convergence/quality evidence. Nine trajectories, identical initialization/data/LR, 72 updates total. No training on test split. No checkpoints recomputed.',
    stop_rule='Stop on any correctness/nonfinite failure. No scaling to long training from a generic gradient difference; direct prior-art overlap already found.',
    prepared_utc=datetime.now(timezone.utc).isoformat(),sources=sources)
(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
(out/'preparation-source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(directory=str(out),sources=sources)))
