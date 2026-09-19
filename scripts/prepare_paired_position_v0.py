"""Freeze within-article position and equal-length evidence-ablation diagnostics."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,tarfile,io
import numpy as np
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    audit=load(R/'results/midpoint64-audit-v2/result.json');assert audit['status']=='verified'
    parent=load(R/'provenance/midpoint64-protocol-v2.json')
    out=R/'data/paired-position-v0';out.mkdir(exist_ok=False)
    tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True);assert tok.is_fast
    enc=lambda s:tok.encode(s,add_special_tokens=False)
    demo=(R/'scripts/prepare_task_quality.py').read_text(encoding='utf-8').split("demonstration='''",1)[1].split("'''",1)[0];prefix=enc(demo)
    oldmeta=load(R/'data/32k-expanded-training-v0/tasks.json');a=np.load(R/'data/32k-expanded-training-v0/tasks.npz');flat=a['input_ids'];offsets=a['offsets']
    newmeta=[];tokens=[];newoff=[0];positions=[.1,.35,.65,.9];replay_count=0
    for index,item in enumerate(oldmeta):
        if item['variant']!='long32768':continue
        old=flat[offsets[index]:offsets[index+1]].tolist();head='\n\nTARGET passage:\n';tail='\n\nEnd TARGET passage.\n\n';s=head+item['target_text']+tail
        evidence=tok(s,add_special_tokens=False,return_offsets_mapping=True);ev=evidence['input_ids'];begin=item['target_token_start'];end=begin+len(ev)
        suffix=enc('\n\nQuestion: '+item['question']+'\n'+'\n'.join(f'{c}. {v}' for c,v in zip('ABCD',item['options']))+'\nAnswer:')
        assert old[:len(prefix)]==prefix and old[begin:end]==ev and old[-len(suffix):]==suffix
        background=old[len(prefix):begin]+old[end:-len(suffix)];assert len(background)+len(prefix)+len(ev)+len(suffix)==32768
        content=[i for i,(l,h) in enumerate(evidence['offset_mapping']) if h>len(head) and l<len(head)+len(item['target_text'])]
        assert content==list(range(content[0],content[-1]+1))
        ablated=ev[:];first,last=content[0],content[-1]+1
        # Replace only article-content-overlapping tokens. Keep evidence markers,
        # total length, filler order, question, choices and answer-token IDs fixed.
        ablated[first:last]=background[:last-first];assert len(ablated)==len(ev) and ablated!=ev
        for pos in positions:
            left=int(len(background)*pos)
            for condition,block in [('present',ev),('ablated',ablated)]:
                seq=prefix+background[:left]+block+background[left:]+suffix;assert len(seq)==32768
                original=condition=='present' and pos==item['evidence_fraction']
                if original:assert seq==old;replay_count+=1
                tokens.extend(seq);newoff.append(len(tokens))
                newmeta.append(dict(item_id=item['item_id'],variant=condition,condition=condition,evidence_fraction=pos,original_replay=original,gold=item['gold'],article_hash=item['article_hash'],target_token_start=len(prefix)+left,replaced_tokens=last-first if condition=='ablated' else 0))
    assert len(newmeta)==512 and replay_count==64 and len({x['article_hash'] for x in newmeta})==64
    np.savez_compressed(out/'tasks.npz',input_ids=np.asarray(tokens,dtype=np.int32),offsets=np.asarray(newoff,dtype=np.int64));save(out/'tasks.json',newmeta)
    save(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_tokens_sha256=sha(R/'data/32k-expanded-training-v0/tasks.npz'),source_metadata_sha256=sha(R/'data/32k-expanded-training-v0/tasks.json'),tasks_sha256=sha(out/'tasks.npz'),metadata_sha256=sha(out/'tasks.json'),articles=64,positions=positions,conditions=['present','ablated'],exact_original_replays=64,scope='Reused development articles. Matched-length distractor substitution, not semantic answer counterfactual. Positions and seeds are repeated measures of the same article. No article selection by outcomes.'))
    jobs=[];files={}
    for j in parent['jobs']:
        if j['step']!=128 or j['k'] not in [0,32]:continue
        j=dict(j);j.update(name=f"paired-k{j['k']}-seed{j['seed']}",phase='paired_position',item_ids=None)
        reference=f"results/midpoint64-stage-v2/trajectory-k{j['k']}-seed{j['seed']}-step128/result.json"
        local=R/'results/cloud-midpoint64-evidence-v2'/reference;j.update(reference=reference,reference_sha256=sha(local));files[reference]=local;jobs.append(j)
    assert len(jobs)==4
    evalsrc=(R/'scripts/eval_midpoint64_v2.py').read_text()
    evalsrc=evalsrc.replace('provenance/midpoint64-protocol-v2.json','provenance/paired-position-protocol-v0.json')
    evalsrc=evalsrc.replace("meta=json.loads((data/'tasks.json').read_text());arrays=np.load(data/'tasks.npz')", "meta=json.loads((R/'data/paired-position-v0/tasks.json').read_text());arrays=np.load(R/'data/paired-position-v0/tasks.npz')")
    evalsrc=evalsrc.replace("if job['phase']=='replay':", "if True:")
    evalsrc=evalsrc.replace("    if reference:\n", "    pred.update(condition=item['condition'],evidence_fraction=item['evidence_fraction'],original_replay=item['original_replay'])\n    if item['original_replay']:\n")
    evalsrc=evalsrc.replace("ref=reference[(item['item_id'],item['variant'])]", "ref=reference[(item['item_id'],'long32768')]")
    evalsrc=evalsrc.replace("expected=128", "expected=512").replace("assert len(predictions)==(8 if job['phase']=='replay' else 128)","assert len(predictions)==512 and sum(x['original_replay'] for x in predictions)==64")
    f=R/'scripts/eval_paired_position_v0.py';ast.parse(evalsrc);f.write_text(evalsrc,encoding='utf-8')
    assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr in ['backward','step'] for x in ast.walk(ast.parse(evalsrc)))
    stage=(R/'scripts/run_midpoint64_stage_v2.py').read_text().replace('midpoint64-protocol-v2','paired-position-protocol-v0').replace('midpoint64-stage-v2','paired-position-stage-v0').replace('midpoint64-queue-v2','paired-position-queue-v0').replace('midpoint64-evidence-v2','paired-position-evidence-v0').replace('eval_midpoint64_v2','eval_paired_position_v0').replace('midpoint64-manifest','paired-position-manifest')
    f=R/'scripts/run_paired_position_stage_v0.py';ast.parse(stage);f.write_text(stage,encoding='utf-8')
    sources={n:h for n,h in parent['source_sha256'].items() if n not in ['scripts/eval_midpoint64_v2.py','scripts/run_midpoint64_stage_v2.py']}
    for n in ['scripts/eval_paired_position_v0.py','scripts/run_paired_position_stage_v0.py','scripts/prepare_paired_position_v0.py']:sources[n]=sha(R/n)
    data=dict(parent['data_sha256'])
    for n in ['tasks.npz','tasks.json','manifest.json']:data['data/paired-position-v0/'+n]=sha(out/n)
    for n,f in files.items():data[n]=sha(f)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),parent_audit_sha256=sha(R/'results/midpoint64-audit-v2/result.json'),jobs=jobs,source_sha256=sources,data_sha256=data,variants=['present','ablated'],evaluation_gpu=parent['evaluation_gpu'],calibration_max_abs_error=1e-6,replay_logit_max_abs_error=1e-6,maximum_seconds=4200,maximum_job_seconds=1000,expected_task_predictions=2048,optimizer_updates=0,primary_estimands=['Paired present-minus-ablated accuracy at each position, dense andK32; average over seeds.','Sparse-minus-dense evidence-use effect: difference of within-article ablation effects.','Present-condition sparse-minus-dense accuracy by position.'],analysis='All64 articles and4 positions; two128-step seeds for each mode. Bootstrap articles jointly across positions, conditions andseeds. Report allposition estimates and article-averaged estimate; exploratory intervals without selecting bestposition. Dense positive evidence effect is a validity diagnostic, not a novelty or noninferiority claim.',scope='Mechanism diagnostic only; no new independent benchmark, no training, no speedup claim. Sparse route causality is not established by a position or ablation effect alone.',maximum_gpu_cost_usd_excluding_setup_storage=4200/3600*.74)
    pp=R/'provenance/paired-position-protocol-v0.json';assert not pp.exists();save(pp,protocol)
    for n in list(sources)+list(data):files.setdefault(n,R/n)
    files[pp.relative_to(R).as_posix()]=pp
    a=R/'exports/paired-position-launch-v0.tar.gz';entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof,predictions=2048,updates=0)))
if __name__=='__main__':main()
