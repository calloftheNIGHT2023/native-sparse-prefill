import hashlib,json,sys
from pathlib import Path
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from run_joint_pilot import sha,save,utc
from prepare_cloud_articles import articles
data=ROOT/'data/cloud-articles-v2'; manifest=json.loads((data/'manifest.json').read_text())
assert sha(data/'tokens.pt')==manifest['tokens_sha256']
tokens=torch.load(data/'tokens.pt',weights_only=True)
seen=set(); previous=set(); counts={}; tokenized={}
for record in manifest['exclusions']:
    p=ROOT/record['path']; assert sha(p)==record['sha256']
    previous.update(e['text_sha256'] for e in json.loads(p.read_text())['examples'])
tokenizer=AutoTokenizer.from_pretrained(ROOT/'data/realtext-v0-r1/assets/model',local_files_only=True)
for split in tokens:
    rows=pq.read_table(ROOT/f'data/realtext-v0-r1/assets/{split}.parquet').column('text').to_pylist()
    boundaries={(start,end) for start,end,_ in articles(rows)}
    examples=[e for e in manifest['examples'] if e['split']==split]
    assert len(examples)==len(tokens[split])
    hashes=set()
    for e,seq in zip(examples,tokens[split]):
        start,end=e['article_start_row'],e['article_end_row_exclusive']
        assert (start,end) in boundaries
        texts=rows[start:end]
        assert not any(hashlib.sha256(t.strip().encode()).hexdigest() in previous for t in texts)
        text='\n'.join(t.strip() for t in texts if t.strip())
        assert hashlib.sha256(text.encode()).hexdigest()==e['article_sha256']
        if e['article_sha256'] not in tokenized:
            tokenized[e['article_sha256']]=tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
        ids=tokenized[e['article_sha256']]; offset=e['token_offset']
        assert seq.tolist()==ids[offset:offset+2049]
        assert hashlib.sha256(json.dumps(seq.tolist()).encode()).hexdigest()==e['token_sha256']
        hashes.add(e['article_sha256'])
    assert not seen.intersection(hashes); seen.update(hashes)
    if split!='train': assert len(hashes)==len(examples)
    counts[split]=dict(windows=len(examples),articles=len(hashes),tokens_per_window=2048)
save(ROOT/'logs/cloud-data-audit.json',dict(utc=utc(),counts=counts,prior_paragraphs=len(previous),
    all_article_boundaries_verified=True,all_tokens_retokenized_and_verified=True,old_articles_excluded=True,
    native_split_articles_disjoint=True,source_sha256=sha(Path(__file__))))
print(json.dumps(counts))
