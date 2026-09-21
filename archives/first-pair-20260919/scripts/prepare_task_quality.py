"""Freeze task IDs, paired contexts and answers before any model evaluation."""
import hashlib,json,random
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=R/'data/task-quality-v0';out.mkdir(exist_ok=False)
 tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True)
 enc=lambda x:tok.encode(x,add_special_tokens=False)
 labels=[enc(' '+x)[0] for x in 'ABCD'];assert all(len(enc(' '+x))==1 for x in 'ABCD')
 src=R/'data/task-quality-sources-v0';source=json.loads((src/'source-manifest.json').read_text())
 for x in source['files']:assert sha(src/(x['split']+'.parquet'))==x['sha256']
 tables={s:pq.read_table(src/(s+'.parquet')).to_pylist() for s in ['train','validation','test']}
 article_hash=lambda x:hashlib.sha256(x.strip().encode()).hexdigest()
 forbidden={article_hash(x['article']) for s in ['validation','test'] for x in tables[s]}
 filler_articles=[];seen=set()
 for x in tables['train']:
  h=article_hash(x['article'])
  if h not in forbidden and h not in seen:seen.add(h);filler_articles.append(x['article'])
 random.Random(2026091570).shuffle(filler_articles)
 filler=enc('\n\n'.join('Unrelated passage:\n'+x for x in filler_articles[:160]));assert len(filler)>32768
 demonstration='''Read the passage and choose the correct option. Answer with one letter: A, B, C, or D.

Example passage: Maya put the blue box in the kitchen.
Question: Where is the blue box?
A. In the garden
B. In the garage
C. In the kitchen
D. In the office
Answer: C

Example passage: The record for silver-forest has value copper.
Question: What is the value for silver-forest?
A. copper
B. velvet
C. marble
D. cedar
Answer: A

Use the passage marked TARGET below for the next question.
'''
 prefix=enc(demonstration);records={s:[] for s in ['pilot','formal']};selections={};used_articles=set()
 def race_items(split,count):
  xs=list(enumerate(tables[split]));random.Random(2026091571 if split=='validation' else 2026091572).shuffle(xs);chosen=[]
  for rownum,x in xs:
   h=article_hash(x['article'])
   if h in used_articles or not (80<=len(enc(x['article']))<=650):continue
   if len(x['options'])!=4 or len(set(x['options']))!=4 or any(len(enc(z))>70 for z in x['options']):continue
   if len(enc(x['question']))>90 or '_' in x['question']:continue
   used_articles.add(h);chosen.append(dict(source_id=x['example_id'],source_row=rownum,article_hash=h,article=x['article'],question=x['question'],options=x['options'],original_gold='ABCD'.index(x['answer'])))
   if len(chosen)==count:return chosen
  raise AssertionError('Not enough distinct eligible passages')
 def variants(split,task,item,i):
  rng=random.Random(2026091580+(0 if split=='pilot' else 10000)+i+(1000 if task=='retrieval' else 0))
  gold=i%4;wrong=[x for j,x in enumerate(item['options']) if j!=item['original_gold']];rng.shuffle(wrong);options=wrong[:];options.insert(gold,item['options'][item['original_gold']])
  suffix='\n\nQuestion: '+item['question']+'\n'+'\n'.join(f'{l}. {x}' for l,x in zip('ABCD',options))+'\nAnswer:'
  suffixids=enc(suffix)
  assert all(enc(suffix+' '+l)==suffixids+[labels[j]] for j,l in enumerate('ABCD'))
  evidence=enc('\n\nTARGET passage:\n'+item['article']+'\n\nEnd TARGET passage.\n\n')
  ids=prefix+evidence+suffixids;assert len(ids)<=1200
  base=dict(item_id=f'{split}-{task}-{i:03}',task=task,gold=gold,options=options,question=item['question'],source_id=item['source_id'],article_hash=item['article_hash'],target_text=item['article'])
  def add(variant,tokenids,start=None):records[split].append(dict(**base,variant=variant,length=len(tokenids),target_token_start=start,target_token_count=len(evidence) if start is not None else 0,input_ids=tokenids))
  add('short',ids,len(prefix));add('no_context',prefix+enc('\nTARGET passage: [not provided]\n')+suffixids)
  if split=='formal':
   fraction=[.1,.5,.9][i%3]
   for length in [8192,16384]:
    n=length-len(prefix)-len(evidence)-len(suffixids);assert n>0
    offset=rng.randrange(len(filler)-n);hay=filler[offset:offset+n];left=int(n*fraction)
    full=prefix+hay[:left]+evidence+hay[left:]+suffixids;assert len(full)==length
    add(f'long{length}',full,len(prefix)+left)
 for split,source_split,count in [('pilot','validation',32),('formal','test',96)]:
  items=race_items(source_split,count);selections[split]=items
  for i,item in enumerate(items):variants(split,'race_mc',item,i)
  for i in range(count):
   key=f'catalog-{split}-{i:04d}'
   values=random.Random(2026091590+i+(10000 if split=='formal' else 0)).sample(['copper','velvet','marble','cedar','silver','cotton','pearl','bronze','linen','granite','amber','willow'],4)
   article='\n'.join(f'The record for {key}-{j} has value {value}.' for j,value in enumerate(values))
   item=dict(source_id=key,article=article,article_hash=article_hash(article),question=f'What is the value for {key}-2?',options=values,original_gold=2)
   variants(split,'retrieval',item,i)
 for split,xs in records.items():
  flat=[];offsets=[0];metadata=[]
  for x in xs:
   d=dict(x);flat.extend(d.pop('input_ids'));offsets.append(len(flat));metadata.append(d)
  np.savez_compressed(out/(split+'.npz'),input_ids=np.asarray(flat,dtype=np.int32),offsets=np.asarray(offsets,dtype=np.int64))
  (out/(split+'.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),model='Qwen2.5-0.5B dense pretrained + locked LoRA checkpoints',label_token_ids=labels,
  tasks=['race_mc','retrieval'],pilot_items_per_task=32,formal_items_per_task=96,pilot_variants=['short','no_context'],formal_variants=['short','no_context','long8192','long16384'],
  gold_balance='Each task/split/variant has exactly 25% A/B/C/D; formal evidence depth 10/50/90% each32, gold-by-depth each8.',
  ability_gate=dict(race_mc=dict(short_min_correct=16,context_gain_min_correct=4),retrieval=dict(short_min_correct=24,context_gain_min_correct=8)),
  gate_checkpoint='dense-seed0 only; pilot-only validation split, no formal task examples opened until gate selected',
  locked_conditions=[[0,2026091560,.0003],[0,2026091561,.0003],[16,2026091560,.001],[16,2026091561,.001]],
  metric='Forced A/B/C/D single-token logit argmax at the end of the prompt; exact choice accuracy. No free-form generation or decode benchmark.',
  timing='Batch1 whole-model prefill plus final-position LM head, CUDA synchronized wall time; excludes CPU tokenization, H2D copy, checkpoint/model load and scoring transfer. Three short warmup forwards excluded. Paired identical prompts. No new training timings.',
  primary='Per task and length, sparse-minus-dense accuracy in percentage points with paired item bootstrap, two seeds; exploratory 5pp margin only if the lower descriptive95% bound exceeds -5pp and dense exceeds chance. Do not infer equivalence from lack of significance.',
  limits='RACE-M adapted marked-target distractor extension, not official RACE or LongBench scores. Synthetic retrieval diagnostic is not official RULER. Base-model task floor must be checked. 96 questions cannot establish tight equivalence; one marked-target paragraph may be easier than natural multi-document QA. Pretraining exposure unknown.',
  source_manifest_sha256=sha(src/'source-manifest.json'),source_revision=source['revision'],
  files={p.name:sha(p) for p in out.iterdir() if p.is_file()},
  references=['https://arxiv.org/abs/1704.04683','https://github.com/NVIDIA/RULER','https://github.com/THUDM/LongBench/blob/main/LongBench/task.md'])
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps(dict(status='prepared',records={k:len(v) for k,v in records.items()},formal_input_tokens=sum(x['length'] for x in records['formal']),hashes=protocol['files'])))
if __name__=='__main__':main()
