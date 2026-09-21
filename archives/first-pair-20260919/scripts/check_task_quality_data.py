"""Independent semantic answer and paired-context integrity checks."""
import json,hashlib,re
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def main():
 d=R/'data/task-quality-v0';p=json.loads((d/'protocol.json').read_text());tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True)
 groups={};counts={}
 for split,rawsplit,n in [('pilot','validation',32),('formal','test',96)]:
  for name in [split+'.json',split+'.npz']:assert hashlib.sha256((d/name).read_bytes()).hexdigest()==p['files'][name]
  raw=pq.read_table(R/f'data/task-quality-sources-v0/{rawsplit}.parquet').to_pylist()
  truth={(hashlib.sha256(x['article'].strip().encode()).hexdigest(),x['question']):x['options']['ABCD'.index(x['answer'])] for x in raw}
  rows=json.loads((d/(split+'.json')).read_text(encoding='utf-8'));arrays=np.load(d/(split+'.npz'));flat=arrays['input_ids'];offsets=arrays['offsets'];g=defaultdict(list)
  assert len(offsets)==len(rows)+1 and offsets[-1]==len(flat)
  for i,x in enumerate(rows):
   tokens=flat[offsets[i]:offsets[i+1]].tolist();assert len(tokens)==x['length'];g[x['item_id']].append(x)
   if x['task']=='race_mc':assert x['options'][x['gold']]==truth[x['article_hash'],x['question']]
   else:
    key=x['question'].removeprefix('What is the value for ').removesuffix('?')
    matches=re.findall(r'The record for (\S+) has value (\w+)\.',x['target_text'])
    assert dict(matches)[key]==x['options'][x['gold']]
   if x['variant']!='no_context':
    evidence=tok.encode('\n\nTARGET passage:\n'+x['target_text']+'\n\nEnd TARGET passage.\n\n',add_special_tokens=False)
    j=x['target_token_start'];assert tokens[j:j+len(evidence)]==evidence
   else:assert x['target_token_count']==0
   if x['variant'].startswith('long'):assert len(tokens)==int(x['variant'][4:])
  for item,variants in g.items():
   assert len({(x['gold'],tuple(x['options']),x['question'],x['target_text']) for x in variants})==1
   assert len({x['variant'] for x in variants})==len(variants)==(2 if split=='pilot' else 4)
  for task in p['tasks']:
   for variant in p[split+'_variants']:
    xs=[x for x in rows if x['task']==task and x['variant']==variant];assert len(xs)==n and Counter(x['gold'] for x in xs)=={i:n//4 for i in range(4)}
  groups[split]={x['article_hash'] for x in rows if x['task']=='race_mc'};counts[split]=len(rows)
 assert not groups['pilot'].intersection(groups['formal'])
 out=R/'results/task-quality-data-audit-v0';out.mkdir(exist_ok=False)
 result=dict(status='passed',records=counts,checks=['source answer retained after option shuffle','retrieval answer matches keyed record','all four labels balanced','paired questions and options identical across lengths','evidence tokens intact','exact 8K/16K lengths','distinct pilot and formal RACE article hashes','frozen file digests'],gpu_updates=0)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
