"""Freeze a minimal direct-word context-use gate before comparing sparse models."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,tarfile,io
import numpy as np
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    assert load(R/'results/option-rotation-audit-v0/result.json')['status']=='verified'
    parent=load(R/'provenance/midpoint64-protocol-v2.json');out=R/'data/word-counterfactual-v0';out.mkdir(exist_ok=False)
    tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True);enc=lambda s:tok.encode(s,add_special_tokens=False)
    words=['apple','copper','marble','velvet'];labels=[enc(' '+x) for x in words];assert all(len(x)==1 for x in labels);labels=[x[0] for x in labels]
    demo='Read the TARGET passage and answer with its secret word.\n\n'
    for word in ['silver','cotton','amber','cedar']:
        demo+=f'TARGET passage: The secret word is {word}.\nQuestion: What is the secret word?\nAnswer: {word}\n\n'
    prefix=enc(demo);suffix=enc('\n\nQuestion: What is the secret word?\nAnswer:')
    old=load(R/'data/32k-expanded-training-v0/tasks.json');a=np.load(R/'data/32k-expanded-training-v0/tasks.npz');xs=a['input_ids'];off=a['offsets'];families=[]
    origdemo=(R/'scripts/prepare_task_quality.py').read_text(encoding='utf-8').split("demonstration='''",1)[1].split("'''",1)[0];origprefix=enc(origdemo)
    for idx,x in enumerate(old):
        if x['variant']!='long32768' or len(families)>=32:continue
        ev=enc('\n\nTARGET passage:\n'+x['target_text']+'\n\nEnd TARGET passage.\n\n');tail=enc('\n\nQuestion: '+x['question']+'\n'+'\n'.join(f'{c}. {v}' for c,v in zip('ABCD',x['options']))+'\nAnswer:')
        tokens=xs[off[idx]:off[idx+1]].tolist();p=x['target_token_start'];assert tokens[p:p+len(ev)]==ev
        bg=tokens[len(origprefix):p]+tokens[p+len(ev):-len(tail)];i=len(families);pos=[.1,.35,.65,.9][i//8]
        # Balanced targets within each position. Exactly one evidence token changes.
        conditions=[]
        for counterfactual in [0,1]:
            gold=(i+counterfactual)%4;value=words[gold];evidence=enc('\n\nTARGET passage: The secret word is '+value+'.\nEnd TARGET passage.\n\n')
            n=32768-len(prefix)-len(evidence)-len(suffix);offset=(i*17)%len(bg);hay=(bg*3)[offset:offset+n];left=int(n*pos)
            conditions.append(dict(gold=gold,evidence=evidence,variants=dict(short=prefix+evidence+suffix,no_context=prefix+enc('\n\nTARGET passage: [not provided]\n\n')+suffix,long32768=prefix+hay[:left]+evidence+hay[left:]+suffix)))
        for variant in ['short','long32768']:
            aa=conditions[0]['variants'][variant];bb=conditions[1]['variants'][variant];assert len(aa)==len(bb) and sum(x!=y for x,y in zip(aa,bb))==1
        assert conditions[0]['variants']['no_context']==conditions[1]['variants']['no_context']
        families.append(dict(family_id=f'word-cf-{i:03}',position=pos,conditions=conditions))
    assert len(families)==32
    metadata=[];flat=[];offsets=[0]
    # Both short conditions are completed before the long ability gate.
    for variant in ['short','no_context','long32768']:
        for family in families:
            for cf,c in enumerate(family['conditions']):
                seq=c['variants'][variant];flat.extend(seq);offsets.append(len(flat));metadata.append(dict(item_id=family['family_id']+f'-cf{cf}',family_id=family['family_id'],counterfactual=cf,variant=variant,gold=c['gold'],value=words[c['gold']],evidence_fraction=family['position'],length=len(seq)))
    assert len(metadata)==192 and all(x['length']==32768 for x in metadata if x['variant']=='long32768')
    np.savez_compressed(out/'tasks.npz',input_ids=np.asarray(flat,dtype=np.int32),offsets=np.asarray(offsets,dtype=np.int64));save(out/'tasks.json',metadata)
    save(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),words=words,label_token_ids=labels,families=32,records=192,source_filler_sha256=sha(R/'data/32k-expanded-training-v0/tasks.npz'),one_token_counterfactual_verified=True,no_context_pair_inputs_identical=True,gold_by_variant=[16,16,16,16],scope='Synthetic assigned-word forced-four-candidate completion; not freegeneration or officialRULER. Each family has paired different factual values; familiar answer words and reused unrelated filler.'))
    job=dict(next(j for j in parent['jobs'] if j['k']==0 and j['step']==0));job.update(name='dense-base-ability',phase='word_counterfactual')
    sources={n:h for n,h in parent['source_sha256'].items() if 'midpoint64_v2' not in n}
    for n in ['scripts/eval_word_counterfactual_v0.py','scripts/run_word_counterfactual_stage_v0.py','scripts/prepare_word_counterfactual_v0.py']:sources[n]=sha(R/n)
    data={n:h for n,h in parent['data_sha256'].items() if not n.endswith('tasks.npz') and not n.endswith('tasks.json')}
    for n in ['tasks.npz','tasks.json','manifest.json']:data['data/word-counterfactual-v0/'+n]=sha(out/n)
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=[job],source_sha256=sources,data_sha256=data,evaluation_gpu=parent['evaluation_gpu'],calibration_max_abs_error=1e-6,answer_token_ids=labels,short_gate=dict(min_correct=56,min_pairs_both_correct=24,min_gain_over_no_context=24),long_gate=dict(min_correct=48,min_pairs_both_correct=20,min_gain_over_no_context=24),maximum_seconds=600,expected_task_predictions_if_short_passes=192,expected_task_predictions_if_short_fails=128,optimizer_updates=0,maximum_gpu_cost_usd_excluding_setup_storage=600/3600*.74,scope='Operational dense-base ability gate only. Evaluate short andno_context first; ifshort gate fails do not score long or any sparse model. Gates frozen before predictions; no template tuning within stage, no independent validation or novelty claim.')
    pp=R/'provenance/word-counterfactual-protocol-v0.json';assert not pp.exists();save(pp,p)
    files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/word-counterfactual-launch-v0.tar.gz';entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
