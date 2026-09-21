"""Pinned WikiText article windows; exclude every previously inspected article."""
import hashlib,json,re
from pathlib import Path
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from run_joint_pilot import utc,sha,save
ROOT=Path(__file__).resolve().parents[1]

def articles(rows):
    start=None
    for i,text in enumerate(rows):
        # WikiText top-level titles have one '=' on each side; sections use '= ='.
        if re.fullmatch(r'=\s+[^=].*?[^=]\s+=',text.strip()):
            if start is not None: yield start,i,rows[start:i]
            start=i
    if start is not None: yield start,len(rows),rows[start:]

def main():
    out=ROOT/'data/cloud-articles-v2'; out.mkdir(parents=True,exist_ok=False)
    started=utc(); assets=ROOT/'data/realtext-v0-r1'
    records=json.loads((assets/'download-manifest.json').read_text())
    for rec in records:
        p=ROOT/rec['path']
        if p.suffix=='.parquet' or p.name.startswith('tokenizer'): assert sha(p)==rec['sha256']
    tokenizer=AutoTokenizer.from_pretrained(assets/'assets/model',local_files_only=True)
    prior_paths=['data/realtext-v0-r1/manifest.json','data/realtext-calibrated-fresh-v0/manifest.json','data/joint-pilot-v0/manifest.json']
    prior_hashes=set(); exclusions=[]
    for path in prior_paths:
        prior=json.loads((ROOT/path).read_text())
        prior_hashes.update(e['text_sha256'] for e in prior['examples'])
        exclusions.append(dict(path=path,sha256=sha(ROOT/path)))
    examples=[]; tensors={}; reports={}; seen_articles=set(); seen_windows=set()
    for split,count in dict(train=384,validation=16,test=13).items():
        rows=pq.read_table(assets/f'assets/{split}.parquet').column('text').to_pylist()
        candidates=[]; excluded=0; article_count=0
        for start,end,texts in articles(rows):
            article_count+=1
            if any(hashlib.sha256(t.strip().encode()).hexdigest() in prior_hashes for t in texts):
                excluded+=1; continue
            text='\n'.join(t.strip() for t in texts if t.strip())
            article_hash=hashlib.sha256(text.encode()).hexdigest()
            if article_hash in seen_articles: continue
            seen_articles.add(article_hash)
            ids=tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
            windows=list(range(0,len(ids)-2048,2048))
            if split!='train' and windows:
                windows=[min(windows,key=lambda o:hashlib.sha256(f'2026091402:{split}:{start}:{o}'.encode()).digest())]
            for offset in windows:
                seq=ids[offset:offset+2049]
                digest=hashlib.sha256(json.dumps(seq).encode()).hexdigest()
                if digest in seen_windows: continue
                seen_windows.add(digest)
                candidates.append((dict(split=split,article_start_row=start,article_end_row_exclusive=end,
                    article_sha256=article_hash,token_offset=offset,token_sha256=digest),seq))
        candidates.sort(key=lambda p:hashlib.sha256(f"2026091402:{split}:{p[0]['token_sha256']}".encode()).digest())
        reports[split]=dict(articles=article_count,excluded_prior_articles=excluded,eligible_windows=len(candidates),requested=count)
        save(out/'preparation-progress.json',reports)
        if len(candidates)<count: raise ValueError(f'Insufficient article-disjoint data: {reports}')
        selected=candidates[:count]
        tensors[split]=torch.tensor([seq for _,seq in selected],dtype=torch.long)
        examples.extend(dict(id=f'{split}-{i:04d}',**rec) for i,(rec,_) in enumerate(selected))
    torch.save(tensors,out/'tokens.pt')
    save(out/'manifest.json',dict(started_utc=started,finished_utc=utc(),sequence_length=2048,
        examples=examples,counts={k:len(v) for k,v in tensors.items()},reports=reports,exclusions=exclusions,
        tokens_sha256=sha(out/'tokens.pt'),script_sha256=sha(Path(__file__)),
        dataset_revision='b08601e04326c79dfdd32d625aee71d232d685c3',
        scope='Native WikiText splits; exclude entire articles containing any of 416 prior paragraphs. Evaluation has one window per distinct article. No model pretraining decontamination claim.'))
    print(json.dumps(reports))
if __name__=='__main__': main()
