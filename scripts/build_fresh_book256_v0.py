"""Build a frozen fresh-book retrieval set, with no model outcomes available."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,urllib.request,base64,random,time
import numpy as np
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def fetch(url):
 for i in range(3):
  try:
   with urllib.request.urlopen(url,timeout=45) as q:return q.read()
  except Exception:
   if i==2:raise
   time.sleep(2)
def main():
 o=R/'data/fresh-book256-v0';o.mkdir(exist_ok=False);(o/'raw').mkdir();started=datetime.now(timezone.utc).isoformat()
 design=dict(created_utc=started,seed=2026091709,books=256,offset=4096,tokens=32768,source='first1000 lexicographic PG19 train GCS objects; deterministic shuffled eligible selection, not representative full-corpus sample',words=['orange','purple','rabbit','planet'],positions=[.1,.35,.65,.9],records=1029,base_gate=dict(short_correct=4,long_correct_min=768),comparisons=['sparse64restored-dense64','sparse128restored-dense128','dense64-sparse128restored'],bootstrap_seed=2026091710,draws=20000,quantiles=[.05/6,1-.05/6],margin_pp=5,scope='New independent book backgrounds with fixed old template and words; four counterfactual facts per book are clustered, not independent. No training, no hypothesis/threshold changes after scores. Old PG19 test books and current training/report token chunks excluded. Unknown base-pretraining exposure. Synthetic four-choice retrieval, not natural QA. Three fixed contrasts, conservative Bonferroni; conditional on two old training seeds. Base gate failure stops unchanged task, not a sparse failure.')
 save(o/'design-before-data.json',design)
 url='https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o?prefix=train%2F&maxResults=1000';raw=fetch(url);(o/'official-train-listing-first1000.json').write_bytes(raw);catalog=json.loads(raw)['items'];assert len(catalog)==1000;random.Random(design['seed']).shuffle(catalog)
 tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True);enc=lambda s:tok.encode(s,add_special_tokens=False)
 old=json.loads((R/'data/pg19-external-v0/official-test-listing.json').read_text());exclude={Path(x['name']).stem for x in old['items']}
 blocked=set()
 for f in ['data/32k-expanded-training-v0/train-calibration.npz','data/32k-expanded-training-v0/report.npz','data/pg19-external-v0/windows.npz']:
  z=np.load(R/f)
  for k in z.files:
   for seq in z[k]:
    for i in range(0,len(seq)-127,64):blocked.add(hashlib.sha256(np.asarray(seq[i:i+128],dtype=np.int32).tobytes()).digest())
 demo='Read the TARGET passage and answer with its secret word.\n\n'
 for w in ['silver','cotton','amber','cedar']:demo+=f'TARGET passage: The secret word is {w}.\nQuestion: What is the secret word?\nAnswer: {w}\n\n'
 prefix=enc(demo);suffix=enc('\n\nQuestion: What is the secret word?\nAnswer:');words=design['words'];evidence=[enc('\n\nTARGET passage: The secret word is '+w+'.\nEnd TARGET passage.\n\n') for w in words];labels=[enc(' '+w) for w in words];assert all(len(x)==1 for x in labels) and len({len(e) for e in evidence})==1
 need=32768-len(prefix)-len(suffix)-len(evidence[0]);families=[];meta=[];parts=[];offsets=[0];attempts=[];contents=set()
 def add(seq,**kw):parts.append(np.asarray(seq,dtype=np.int32));offsets.append(offsets[-1]+len(seq));meta.append(dict(**kw,length=len(seq)))
 for g,e in enumerate(evidence):add(prefix+e+suffix,item_id=f'short-{g}',family_id=None,variant='short',gold=g,value=words[g])
 add(prefix+enc('\n\nTARGET passage: [not provided]\n\n')+suffix,item_id='no-context',family_id=None,variant='no_context',gold=None,value=None)
 for obj in catalog:
  if len(families)==256:break
  bid=Path(obj['name']).stem
  if bid in exclude or int(obj['size'])<100000:continue
  b=fetch(obj['mediaLink']);assert len(b)==int(obj['size']) and base64.b64encode(hashlib.md5(b).digest()).decode()==obj['md5Hash'];h=hashlib.sha256(b).hexdigest();f=o/'raw'/f'{bid}.txt';f.write_bytes(b)
  tokens=enc(b.decode('utf-8'));row=dict(book_id=bid,object_name=obj['name'],generation=obj['generation'],md5=obj['md5Hash'],sha256=h,bytes=len(b),token_count=len(tokens))
  if h in contents or len(tokens)<4096+need:row['status']='ineligible_duplicate_or_short';attempts.append(row);continue
  bg=tokens[4096:4096+need]
  overlaps=sum(hashlib.sha256(np.asarray(bg[i:i+128],dtype=np.int32).tobytes()).digest() in blocked for i in range(0,len(bg)-127,64))
  if overlaps:row.update(status='rejected_exact_chunk_overlap',overlaps=overlaps);attempts.append(row);continue
  i=len(families);pos=design['positions'][i%4];left=int(need*pos);family=f'book256-{i:03}';row.update(status='selected',family_id=family,position=pos,background_sha256=hashlib.sha256(np.asarray(bg,dtype=np.int32).tobytes()).hexdigest());families.append(row);contents.add(h);attempts.append(row);seqs=[]
  for g,e in enumerate(evidence):
   seq=prefix+bg[:left]+e+bg[left:]+suffix;assert len(seq)==32768;seqs.append(np.asarray(seq,dtype=np.int32));add(seq,item_id=f'{family}-value{g}',family_id=family,variant='long32768',gold=g,value=words[g],position=pos,book_id=bid)
  assert all(np.count_nonzero(seqs[0]!=x)==1 for x in seqs[1:])
  if len(families)%16==0:print(json.dumps(dict(books=len(families),attempted=len(attempts))),flush=True)
 assert len(families)==256 and len(meta)==1029,'Insufficient books: preserve incomplete build; do not silently reduce n'
 np.savez_compressed(o/'tasks.npz',input_ids=np.concatenate(parts),offsets=np.asarray(offsets,dtype=np.int64));save(o/'tasks.json',meta)
 manifest=dict(design=design,created_utc=datetime.now(timezone.utc).isoformat(),builder_sha256=sha(Path(__file__)),label_token_ids=[x[0] for x in labels],families=families,attempts=attempts,source_url=url,listing_sha256=sha(o/'official-train-listing-first1000.json'),tasks_sha256=sha(o/'tasks.npz'),meta_sha256=sha(o/'tasks.json'),scope=design['scope']);save(o/'manifest.json',manifest);print(json.dumps(dict(status='built',books=256,records=1029,manifest_sha256=sha(o/'manifest.json'))))
if __name__=='__main__':main()
