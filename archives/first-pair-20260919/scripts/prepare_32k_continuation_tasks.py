"""Freeze corpus splits and unseen-in-this-project RACE items before training."""
import hashlib,json,random,requests
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 out=R/'data/32k-continuation-tasks-v0';out.mkdir(exist_ok=False)
 tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True)
 enc=lambda x:tok.encode(x,add_special_tokens=False)
 src=R/'data/task-quality-sources-v0';source=json.loads((src/'source-manifest.json').read_text())
 for x in source['files']:assert sha(src/(x['split']+'.parquet'))==x['sha256']
 tables={s:pq.read_table(src/(s+'.parquet')).to_pylist() for s in ['train','validation','test']}
 ah=lambda x:hashlib.sha256(x.strip().encode()).hexdigest()
 previous={x['article_hash'] for n in ['pilot','formal'] for x in json.loads((R/f'data/task-quality-v0/{n}.json').read_text(encoding='utf-8')) if x['task']=='race_mc'}
 previous.update(x['article_hash'] for x in json.loads((R/'data/32k-adaptation-v0/tasks.json').read_text(encoding='utf-8')))
 forbidden={ah(x['article']) for s in ['validation','test'] for x in tables[s]};fill=[];seen=set()
 for x in tables['train']:
  h=ah(x['article'])
  if h not in forbidden and h not in seen:seen.add(h);fill.append(x['article'])
 random.Random(2026091680).shuffle(fill);filler=enc('\n\n'.join('Unrelated passage:\n'+x for x in fill[:400]));assert len(filler)>65536
 # Reuse the original prompt wording, only IDs, unseen passages and context length change.
 script=(R/'scripts/prepare_task_quality.py').read_text(encoding='utf-8');demonstration=script.split("demonstration='''",1)[1].split("'''",1)[0];prefix=enc(demonstration)
 xs=list(enumerate(tables['test']));random.Random(2026091681).shuffle(xs);selected=[];used=set(previous)
 for num,x in xs:
  h=ah(x['article'])
  if h in used or not 80<=len(enc(x['article']))<=650:continue
  if len(x['options'])!=4 or len(set(x['options']))!=4 or any(len(enc(v))>70 for v in x['options']):continue
  if len(enc(x['question']))>90 or '_' in x['question']:continue
  used.add(h);selected.append((num,x,h))
  if len(selected)==64:break
 assert len(selected)==64;labels=[enc(' '+c)[0] for c in 'ABCD'];assert all(len(enc(' '+c))==1 for c in 'ABCD')
 flat=[];offsets=[0];meta=[]
 for i,(num,x,h) in enumerate(selected):
  rr=random.Random(2026091682+i);original='ABCD'.index(x['answer']);gold=i%4;wrong=[v for j,v in enumerate(x['options']) if j!=original];rr.shuffle(wrong);options=wrong[:];options.insert(gold,x['options'][original])
  suffix=enc('\n\nQuestion: '+x['question']+'\n'+'\n'.join(f'{c}. {v}' for c,v in zip('ABCD',options))+'\nAnswer:')
  evidence=enc('\n\nTARGET passage:\n'+x['article']+'\n\nEnd TARGET passage.\n\n')
  short=prefix+evidence+suffix;assert len(short)<=1200
  n=32768-len(prefix)-len(evidence)-len(suffix);fraction=[.1,.35,.65,.9][(i//4)%4];left=int(n*fraction);offset=rr.randrange(len(filler)-n);hay=filler[offset:offset+n]
  long=prefix+hay[:left]+evidence+hay[left:]+suffix;assert len(long)==32768
  for variant,tokens in [('short',short),('no_context',prefix+enc('\nTARGET passage: [not provided]\n')+suffix),('long32768',long)]:
   flat.extend(tokens);offsets.append(len(flat));meta.append(dict(item_id=f'32k-cont-race-{i:03}',variant=variant,length=len(tokens),gold=gold,source_row=num,source_id=x['example_id'],article_hash=h,target_text=x['article'],question=x['question'],options=options,evidence_fraction=fraction,target_token_start=len(prefix)+left if variant=='long32768' else (len(prefix) if variant=='short' else None)))
 assert {h for _,_,h in selected}.isdisjoint(previous)
 np.savez_compressed(out/'tasks.npz',input_ids=np.asarray(flat,dtype=np.int32),offsets=np.asarray(offsets,dtype=np.int64));dump(out/'tasks.json',meta)
 dump(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),excluded_article_hashes=sorted(previous),source_manifest_sha256=sha(src/'source-manifest.json'),task_metadata_sha256=sha(out/'tasks.json'),task_tokens_sha256=sha(out/'tasks.npz'),label_token_ids=labels,scope='64 new project-unseen RACE test articles; pretrained contamination unknown; marked-target extension is not official RACE. Frozen before continuation outcomes.'))
 (out/'preparation-source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps(dict(status='prepared',items=64,variants=192)))
if __name__=='__main__':main()
