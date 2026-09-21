"""Freeze all four cyclic option orders; no best-order or best-item selection."""
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
    assert load(R/'results/paired-position-audit-v0/result.json')['status']=='verified'
    parent=load(R/'provenance/paired-position-protocol-v0.json')
    out=R/'data/option-rotation-v0';out.mkdir(exist_ok=False)
    tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True);enc=lambda s:tok.encode(s,add_special_tokens=False)
    meta=load(R/'data/32k-expanded-training-v0/tasks.json');a=np.load(R/'data/32k-expanded-training-v0/tasks.npz');flat=a['input_ids'];offsets=a['offsets']
    newmeta=[];tokens=[];newoff=[0]
    def suffix(question,options):return enc('\n\nQuestion: '+question+'\n'+'\n'.join(f'{c}. {v}' for c,v in zip('ABCD',options))+'\nAnswer:')
    for index,item in enumerate(meta):
        if item['variant'] not in ['short','long32768']:continue
        old=flat[offsets[index]:offsets[index+1]].tolist();end=suffix(item['question'],item['options']);assert old[-len(end):]==end
        for rotation in range(4):
            options=item['options'][rotation:]+item['options'][:rotation];newend=suffix(item['question'],options)
            assert len(newend)==len(end),'No length edits allowed to isolate option order'
            seq=old[:-len(end)]+newend;assert len(seq)==len(old)
            if rotation==0:assert seq==old
            tokens.extend(seq);newoff.append(len(tokens));gold=(item['gold']-rotation)%4
            assert options[gold]==item['options'][item['gold']]
            newmeta.append(dict(item_id=item['item_id'],variant=item['variant'],rotation=rotation,original_gold=item['gold'],gold=gold,original_replay=rotation==0 and item['variant']=='long32768',article_hash=item['article_hash'],length=len(seq),options=options))
    assert len(newmeta)==512 and sum(x['original_replay'] for x in newmeta)==64
    for v in ['short','long32768']:
        assert all(sum(x['variant']==v and x['gold']==g for x in newmeta)==64 for g in range(4))
    np.savez_compressed(out/'tasks.npz',input_ids=np.asarray(tokens,dtype=np.int32),offsets=np.asarray(newoff,dtype=np.int64));save(out/'tasks.json',newmeta)
    save(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_tokens_sha256=sha(R/'data/32k-expanded-training-v0/tasks.npz'),source_metadata_sha256=sha(R/'data/32k-expanded-training-v0/tasks.json'),tasks_sha256=sha(out/'tasks.npz'),metadata_sha256=sha(out/'tasks.json'),articles=64,rotations=[0,1,2,3],variants=['short','long32768'],lengths_unchanged=True,scope='Reused64 development articles; each option occupies each letter once. Prefix, passage, filler andquestion tokens untouched. No outcome-based order choice, calibration fit, or synthetic retrieval claim.'))
    jobs=[]
    for j in parent['jobs']:
        j=dict(j);j.update(name=f"rotation-k{j['k']}-seed{j['seed']}",phase='option_rotation');jobs.append(j)
    s=(R/'scripts/eval_paired_position_v0.py').read_text().replace('paired-position-protocol-v0','option-rotation-protocol-v0').replace('data/paired-position-v0','data/option-rotation-v0')
    s=s.replace("pred.update(condition=item['condition'],evidence_fraction=item['evidence_fraction'],original_replay=item['original_replay'])", "pred.update(rotation=item['rotation'],original_gold=item['original_gold'],semantic_prediction=(guess+item['rotation'])%4,original_replay=item['original_replay'])")
    f=R/'scripts/eval_option_rotation_v0.py';ast.parse(s);f.write_text(s,encoding='utf-8')
    assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr in ['backward','step'] for x in ast.walk(ast.parse(s)))
    s=(R/'scripts/run_paired_position_stage_v0.py').read_text().replace('paired-position','option-rotation').replace('eval_paired_position_v0','eval_option_rotation_v0')
    f=R/'scripts/run_option_rotation_stage_v0.py';ast.parse(s);f.write_text(s,encoding='utf-8')
    sources={n:h for n,h in parent['source_sha256'].items() if 'paired_position' not in n}
    for n in ['scripts/eval_option_rotation_v0.py','scripts/run_option_rotation_stage_v0.py','scripts/prepare_option_rotation_v0.py']:sources[n]=sha(R/n)
    data={n:h for n,h in parent['data_sha256'].items() if 'paired-position-v0' not in n}
    for n in ['tasks.npz','tasks.json','manifest.json']:data['data/option-rotation-v0/'+n]=sha(out/n)
    protocol=dict(parent);protocol.update(created_utc=datetime.now(timezone.utc).isoformat(),parent_audit_sha256=sha(R/'results/paired-position-audit-v0/result.json'),jobs=jobs,source_sha256=sources,data_sha256=data,variants=['short','long32768'],maximum_seconds=2400,maximum_job_seconds=550,expected_task_predictions=2048,maximum_gpu_cost_usd_excluding_setup_storage=2400/3600*.74,primary_estimands=['Uniform-four-order accuracy, original-order accuracy, sparse-minus-dense differences and change in those differences, short and32K separately.','Semantic answer consistency over allfour rotations; raw letter frequencies.'],analysis='Bootstrap64 articles jointly across allorders andtwo seeds. Allrotations equally weighted. No selecting orders, averaging logits for a favorable ensemble, or fitting label-bias correction. Conditional exploratory intervals.',scope='Known MCQA order-bias diagnostic applied to sparse-trained checkpoints; not claimed as novel. No newtraining orindependent confirmation; no equivalent-quality orspeed claim.')
    pp=R/'provenance/option-rotation-protocol-v0.json';assert not pp.exists();save(pp,protocol)
    files={}
    for n in list(sources)+list(data):files[n]=R/'results/cloud-midpoint64-evidence-v2'/n if n.startswith('results/midpoint64-stage-v2/') else R/n
    files[pp.relative_to(R).as_posix()]=pp
    a=R/'exports/option-rotation-launch-v0.tar.gz';entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof,predictions=2048,updates=0)))
if __name__=='__main__':main()
