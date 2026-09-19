"""Freeze new, article-disjoint backgrounds and four new factual values for common dense evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,random,tarfile,io
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ah(s):return hashlib.sha256(s.strip().encode()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    out=R/'data/fresh-word-dense-v0';out.mkdir(exist_ok=False)
    tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True);enc=lambda s:tok.encode(s,add_special_tokens=False)
    src=R/'data/task-quality-sources-v0';sm=load(src/'source-manifest.json')
    for x in sm['files']:assert sha(src/(x['split']+'.parquet'))==x['sha256']
    tables={s:pq.read_table(src/(s+'.parquet')).to_pylist() for s in ['train','validation','test']}
    forbidden={ah(x['article']) for s in ['validation','test'] for x in tables[s]};seen=set();fill=[]
    for x in tables['train']:
        h=ah(x['article'])
        if h not in forbidden and h not in seen:seen.add(h);fill.append(x['article'])
    prior=set()
    recipes=[(2026091570,160),(2026091670,400),(2026091680,400),(2026091690,400)]
    for seed,n in recipes:
        z=fill[:];random.Random(seed).shuffle(z);prior.update(ah(s) for s in z[:n])
    training_text=(R/'data/flashmoba-pool-training-v0/train.txt').read_text(encoding='utf-8')
    normalized_train=' '.join(training_text.split());exact_train_overlap=[];fresh=[]
    for s in fill:
        if ah(s) in prior:continue
        if ' '.join(s.split()) in normalized_train:exact_train_overlap.append(ah(s));continue
        fresh.append(s)
    random.Random(2026091672).shuffle(fresh)
    words=['orange','purple','rabbit','planet'];labels=[enc(' '+w) for w in words];assert all(len(x)==1 for x in labels);labels=[x[0] for x in labels]
    demo='Read the TARGET passage and answer with its secret word.\n\n'
    for w in ['silver','cotton','amber','cedar']:demo+=f'TARGET passage: The secret word is {w}.\nQuestion: What is the secret word?\nAnswer: {w}\n\n'
    prefix=enc(demo);suffix=enc('\n\nQuestion: What is the secret word?\nAnswer:')
    evidence=[enc('\n\nTARGET passage: The secret word is '+w+'.\nEnd TARGET passage.\n\n') for w in words]
    assert len({len(x) for x in evidence})==1 and all(sum(a!=b for a,b in zip(evidence[0],e))==1 for e in evidence[1:])
    required=32768-len(prefix)-len(evidence[0])-len(suffix);families=[];cursor=0;all_used=set()
    for i in range(32):
        bg=[];articles=[]
        while len(bg)<required:
            assert cursor<len(fresh),'Not enough disjoint text; do not recycle'
            text=fresh[cursor];cursor+=1;h=ah(text);assert h not in prior and h not in all_used;all_used.add(h)
            tokens=enc('Unrelated passage:\n'+text+'\n\n');articles.append(dict(article_sha256=h,tokens=len(tokens)));bg.extend(tokens)
        bg=bg[:required];families.append(dict(family_id=f'fresh-word-{i:03}',position=[.1,.35,.65,.9][i%4],background=bg,background_sha256=hashlib.sha256(np.asarray(bg,dtype=np.int32).tobytes()).hexdigest(),articles=articles))
    assert len(families)==32 and len({f['background_sha256'] for f in families})==32
    meta=[];flat=[];offsets=[0]
    def add(seq,**kw):
        flat.extend(seq);offsets.append(len(flat));meta.append(dict(**kw,length=len(seq)))
    for gold,e in enumerate(evidence):add(prefix+e+suffix,item_id=f'short-{gold}',family_id=None,variant='short',gold=gold,value=words[gold])
    add(prefix+enc('\n\nTARGET passage: [not provided]\n\n')+suffix,item_id='no-context',family_id=None,variant='no_context',gold=None,value=None)
    for f in families:
        bg=f['background'];left=int(required*f['position']);seqs=[]
        for gold,e in enumerate(evidence):
            seq=prefix+bg[:left]+e+bg[left:]+suffix;assert len(seq)==32768;seqs.append(seq)
            add(seq,item_id=f['family_id']+f'-value{gold}',family_id=f['family_id'],variant='long32768',gold=gold,value=words[gold],position=f['position'])
        assert all(sum(a!=b for a,b in zip(seqs[0],s))==1 for s in seqs[1:])
    assert len(meta)==133
    np.savez_compressed(out/'tasks.npz',input_ids=np.asarray(flat,dtype=np.int32),offsets=np.asarray(offsets,dtype=np.int64));save(out/'tasks.json',meta)
    histories=['prepare_task_quality.py','prepare_32k_adaptation.py','prepare_32k_continuation_tasks.py','prepare_fresh256_confirmation.py']
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),words=words,label_token_ids=labels,families=32,long_records=128,records=133,article_count=len(all_used),prior_exclusion_recipes=recipes,prior_article_hashes=sorted(prior),excluded_exact_training_articles=exact_train_overlap,training_text_sha256=sha(R/'data/flashmoba-pool-training-v0/train.txt'),source_manifest_sha256=sha(src/'source-manifest.json'),historical_preparer_sha256={n:sha(R/'scripts'/n) for n in histories},backgrounds=[{k:v for k,v in f.items() if k!='background'} for f in families],scope='Fresh relative to four documented prior RACE background pools; articles disjoint across32 backgrounds, no tiling. Excludes normalized whole-article matches in current WikiText training text; no claim of unknown pretraining exposure absence or near-duplicate exclusion. Same synthetic question template, new candidate words and new background articles. All four values per background, each differing by exactly one evidence token. Four unique short prompts and one unique no-context prompt only. Not natural QA, free generation or official RULER.')
    save(out/'manifest.json',manifest)
    old=load(R/'provenance/midpoint64-protocol-v2.json');base=dict(next(j for j in old['jobs'] if j['k']==0 and j['step']==0));base.update(name='base-ability',phase='base_gate');jobs=[base]
    for k,seed in [(0,2026091660),(32,2026091660),(32,2026091661),(0,2026091661)]:
        j=dict(next(j for j in old['jobs'] if j['k']==k and j['seed']==seed and j['step']==128));j.update(name=f'common-dense-train{k}-seed{seed}',phase='confirmation');jobs.append(j)
    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','eval_fresh_word_dense_v0.py','run_fresh_word_dense_stage_v0.py','prepare_fresh_word_dense_v0.py','report_fresh_word_dense_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in names+histories}
    data={n:sha(R/n) for n in ['data/32k-expanded-training-v0/config.json','data/32k-expanded-training-v0/train-calibration.npz','data/task-quality-sources-v0/source-manifest.json']}
    for n in ['tasks.npz','tasks.json','manifest.json']:data['data/fresh-word-dense-v0/'+n]=sha(out/n)
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_sha256=sources,data_sha256=data,evaluation_gpu=old['evaluation_gpu'],calibration_max_abs_error=1e-6,answer_token_ids=labels,base_long_min_correct=96,expected_task_predictions=665,expected_gradient_passes=0,optimizer_updates=0,maximum_seconds=1500,maximum_job_seconds=290,maximum_gpu_cost_usd_excluding_setup_storage=1500*.74/3600,primary=dict(checkpoint_step=128,evaluation_k=0,accuracy_noninferiority_margin_pp=5,method='Average both paired training seeds within each of32 background families; resample backgrounds jointly across all4values and2seeds,20000 draws seed2026091673. Report lower two-sided95% bound; pass only if lower>-5pp. Report each seed and family-all-four accuracy. No model/checkpoint/word selection after outcomes.'),scope=manifest['scope']+' Operational gate on untrained base:4/4 short and>=96/128 long; otherwise stop without changing task or evaluating trained comparisons. Native calibration replay1e-6 before common-dense attention. Base ability gating does not establish general LLM ability. Two reused training seeds; uncertainty conditional on them and synthetic task. No suffix protection or inference speed objective; no new weights or training.')
    pp=R/'provenance/fresh-word-dense-protocol-v0.json';assert not pp.exists();save(pp,p)
    files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/fresh-word-dense-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',articles=len(all_used),protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
