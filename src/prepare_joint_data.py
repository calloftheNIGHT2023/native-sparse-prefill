"""Token-only fresh paragraphs from pinned local assets; no model execution."""
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj): path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def utc(): return datetime.now(timezone.utc).isoformat()

def main(args):
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    started=utc(); assets=ROOT/cfg['asset_source']
    records=json.loads((assets/'download-manifest.json').read_text(encoding='utf-8'))
    used=[]
    for rec in records:
        path=ROOT/rec['path']
        if path.suffix=='.parquet' or path.name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
            assert sha(path)==rec['sha256'],str(path)
            used.append(rec)
    tokenizer=AutoTokenizer.from_pretrained(assets/'assets/model',local_files_only=True,trust_remote_code=False)
    seen_text=set(); seen_prefix=set(); excluded=[]
    for path in cfg['exclude_examples_from']:
        prior=json.loads((ROOT/path).read_text(encoding='utf-8'))
        seen_text.update(x['text_sha256'] for x in prior['examples'])
        seen_prefix.update(x['token_prefix_sha256'] for x in prior['examples'])
        excluded.append({'path':path,'sha256':sha(ROOT/path),'count':len(prior['examples'])})
    examples=[]; tensors={}
    for split,count in cfg['examples'].items():
        rows=pq.read_table(assets/f'assets/{split}.parquet',columns=['text']).column('text').to_pylist()
        order=sorted(range(len(rows)),key=lambda i:hashlib.sha256(f"{cfg['selection_seed']}:{split}:{i}".encode()).digest())
        sequences=[]
        for row_idx in order:
            text=rows[row_idx].strip(); text_hash=hashlib.sha256(text.encode()).hexdigest()
            if not text or text.startswith('=') or text_hash in seen_text: continue
            ids=tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
            if len(ids)<cfg['sequence_length']+1: continue
            ids=ids[:cfg['sequence_length']+1]
            prefix=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if prefix in seen_prefix: continue
            seen_text.add(text_hash); seen_prefix.add(prefix)
            examples.append({'id':f'{split}-{len(sequences):04d}','split':split,'row_index':row_idx,
                'text_sha256':text_hash,'token_prefix_sha256':prefix})
            sequences.append(ids)
            if len(sequences)==count: break
        if len(sequences)!=count: raise ValueError(f'Insufficient {split} paragraphs: {len(sequences)}/{count}')
        tensors[split]=torch.tensor(sequences)
    torch.save(tensors,out/'tokens.pt')
    save(out/'frozen-config.json',cfg)
    save(out/'manifest.json',{'started_utc':started,'finished_utc':utc(),'examples':examples,'assets':used,
        'exclusions':excluded,'tokens_sha256':sha(out/'tokens.pt'),'config_sha256':sha(args.config),
        'scope':'Fresh exact-disjoint paragraphs, native WikiText splits; no model pretraining decontamination claim'})
    print(json.dumps({'counts':{k:len(v) for k,v in tensors.items()},'started_utc':started,'finished_utc':utc()}))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

