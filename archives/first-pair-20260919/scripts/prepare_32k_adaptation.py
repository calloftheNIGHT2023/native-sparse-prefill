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
 out=R/'data/32k-adaptation-v0';out.mkdir(exist_ok=False)
 old=json.loads((R/'data/flashmoba-amp-recovery-v0/config.json').read_text());tok=AutoTokenizer.from_pretrained(R/old['model_path'],local_files_only=True)
 enc=lambda x:tok.encode(x,add_special_tokens=False)
 streams={};sources=[]
 for split in ['train','validation']:
  p=R/f'data/flashmoba-pool-training-v0/{split}.txt';assert sha(p)==next(x['sha256'] for x in old['sources'] if x['split']==split)
  streams[split]=np.asarray(enc(p.read_text(encoding='utf-8')),dtype=np.int64);sources.append(dict(split=split,path=p.relative_to(R).as_posix(),sha256=sha(p),tokens=len(streams[split])))
 url=f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{old['dataset_revision']}/wikitext-2-raw-v1/test-00000-of-00001.parquet"
 res=requests.get(url,timeout=60);res.raise_for_status();p=out/'test-source.parquet';p.write_bytes(res.content)
 assert sha(p)=='5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
 texts=pq.read_table(p)['text'].to_pylist();text='\n'.join(texts);(out/'test-source.txt').write_text(text,encoding='utf-8')
 streams['test']=np.asarray(enc(text),dtype=np.int64);sources.append(dict(split='test',url=url,sha256=sha(p),text_sha256=sha(out/'test-source.txt'),tokens=len(streams['test'])))
 width=32769;rng=np.random.default_rng(2026091669)
 selection=dict(train=rng.choice(len(streams['train'])//width,32,replace=False).tolist(),calibration=list(range(4)),report=list(range(len(streams['test'])//width)))
 assert len(selection['report'])>=6
 def windows(split,indices):return np.stack([streams[split][i*width:(i+1)*width] for i in indices])
 train=windows('train',selection['train']);cal=windows('validation',selection['calibration']);report=windows('test',selection['report'])
 # Exact full-window contamination is excluded; pretrained and historical corpus exposure are not.
 hashes=lambda arr:{hashlib.sha256(x.tobytes()).hexdigest() for x in arr}
 assert hashes(train).isdisjoint(hashes(cal)|hashes(report)) and hashes(cal).isdisjoint(hashes(report))
 np.savez(out/'train-calibration.npz',train=train,calibration=cal);np.savez(out/'report.npz',report=report)
 src=R/'data/task-quality-sources-v0';source=json.loads((src/'source-manifest.json').read_text())
 for x in source['files']:assert sha(src/(x['split']+'.parquet'))==x['sha256']
 tables={s:pq.read_table(src/(s+'.parquet')).to_pylist() for s in ['train','validation','test']}
 ah=lambda x:hashlib.sha256(x.strip().encode()).hexdigest()
 previous={x['article_hash'] for n in ['pilot','formal'] for x in json.loads((R/f'data/task-quality-v0/{n}.json').read_text(encoding='utf-8')) if x['task']=='race_mc'}
 forbidden={ah(x['article']) for s in ['validation','test'] for x in tables[s]};fill=[];seen=set()
 for x in tables['train']:
  h=ah(x['article'])
  if h not in forbidden and h not in seen:seen.add(h);fill.append(x['article'])
 random.Random(2026091670).shuffle(fill);filler=enc('\n\n'.join('Unrelated passage:\n'+x for x in fill[:400]));assert len(filler)>65536
 # Reuse the original prompt wording, only IDs, unseen passages and context length change.
 script=(R/'scripts/prepare_task_quality.py').read_text(encoding='utf-8');demonstration=script.split("demonstration='''",1)[1].split("'''",1)[0];prefix=enc(demonstration)
 xs=list(enumerate(tables['test']));random.Random(2026091671).shuffle(xs);selected=[];used=set(previous)
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
  rr=random.Random(2026091672+i);original='ABCD'.index(x['answer']);gold=i%4;wrong=[v for j,v in enumerate(x['options']) if j!=original];rr.shuffle(wrong);options=wrong[:];options.insert(gold,x['options'][original])
  suffix=enc('\n\nQuestion: '+x['question']+'\n'+'\n'.join(f'{c}. {v}' for c,v in zip('ABCD',options))+'\nAnswer:')
  evidence=enc('\n\nTARGET passage:\n'+x['article']+'\n\nEnd TARGET passage.\n\n')
  short=prefix+evidence+suffix;assert len(short)<=1200
  n=32768-len(prefix)-len(evidence)-len(suffix);fraction=[.1,.35,.65,.9][(i//4)%4];left=int(n*fraction);offset=rr.randrange(len(filler)-n);hay=filler[offset:offset+n]
  long=prefix+hay[:left]+evidence+hay[left:]+suffix;assert len(long)==32768
  for variant,tokens in [('short',short),('no_context',prefix+enc('\nTARGET passage: [not provided]\n')+suffix),('long32768',long)]:
   flat.extend(tokens);offsets.append(len(flat));meta.append(dict(item_id=f'32k-race-{i:03}',variant=variant,length=len(tokens),gold=gold,source_row=num,source_id=x['example_id'],article_hash=h,target_text=x['article'],question=x['question'],options=options,evidence_fraction=fraction,target_token_start=len(prefix)+left if variant=='long32768' else (len(prefix) if variant=='short' else None)))
 assert {h for _,_,h in selected}.isdisjoint(previous)
 np.savez_compressed(out/'tasks.npz',input_ids=np.asarray(flat,dtype=np.int32),offsets=np.asarray(offsets,dtype=np.int64));dump(out/'tasks.json',meta)
 cfg={k:old[k] for k in ['model_path','model_revision','dataset_revision','chunk_size','lora_rank','lora_alpha','betas','eps','weight_decay','gradient_clip','extension_sha256','deterministic_backward']}
 cfg.update(version=0,prepared_utc=datetime.now(timezone.utc).isoformat(),length=32768,steps=64,warmup_steps=4,methods=[0,32],learning_rates=[.0003,.001],seeds=[2026091660,2026091661],calibration_steps=[0,16,32,64],checkpoint_steps=[0,2,16,32,64],selection=selection,sources=sources,data_sha256=sha(out/'train-calibration.npz'),report_sha256=sha(out/'report.npz'),task_metadata_sha256=sha(out/'tasks.json'),task_tokens_sha256=sha(out/'tasks.npz'),label_token_ids=labels,task_source_manifest_sha256=sha(src/'source-manifest.json'),excluded_old_task_article_hashes=sorted(previous),maximum_controller_seconds=3600,
 selection_rule='Same two LR candidates for each method; choose minimum seed0 calibration NLL at step64, ties lower LR. Repeat chosen LR on seed1. Only then evaluate selected four models and seed0 step0 baselines on fixed report and tasks. No quality-superiority requirement.',
 scope='Dense-pretrained Qwen2.5-0.5B LoRA, 32K WikiText packed corpus adaptation, not full-parameter native sparse pretraining. Official WikiText test split separated from train/calibration; pretrained and historical corpus contamination not excluded. New 64 RACE test passages exclude previous project pilot/formal articles, marked-target extension not official benchmark. Fixed-token cost and approximate quality tradeoff; two shared-order initialization seeds. No statistical equivalence or novelty claim.')
 cfg['sources_sha256']={n:sha(R/'scripts'/n) for n in ['run_32k_adaptation.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py']}
 dump(out/'config.json',cfg);(out/'preparation-source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps(dict(status='prepared',train_shape=list(train.shape),calibration_shape=list(cal.shape),report_shape=list(report.shape),task_items=64,task_predictions_per_model=192,config_sha256=sha(out/'config.json'))))
if __name__=='__main__':main()
