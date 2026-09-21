"""Freeze privileged fixed-budget routing diagnosis; no new method claim."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile,io
import numpy as np
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    assert load(R/'logs/word-route-preflight-v0-cloud.json')['status']=='passed'
    pre=load(R/'logs/word-route-preflight-v0-cloud.json')
    assert pre['route_sha256']==sha(R/'scripts/word_route_intervention.py')
    assert load(R/'results/word-operator-swap-audit-v0/result.json')['status']=='verified'
    old=load(R/'provenance/word-operator-swap-protocol-v0.json')
    tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True)
    suffix=tok.encode('\n\nQuestion: What is the secret word?\nAnswer:',add_special_tokens=False)
    meta=load(R/'data/word-counterfactual-v0/tasks.json');a=np.load(R/'data/word-counterfactual-v0/tasks.npz');flat=a['input_ids'];off=a['offsets']
    groups={};placements={}
    for i,x in enumerate(meta):
        if x['variant']=='long32768':groups.setdefault(x['family_id'],[]).append((x,flat[off[i]:off[i+1]]))
    assert len(groups)==32
    for group in groups.values():
        assert len(group)==2
        diff=np.flatnonzero(group[0][1]!=group[1][1]);assert len(diff)==1
        pos=int(diff[0]);target=pos//128;sham=target+2;start=32768-len(suffix)
        assert 0<=target<sham<start//128
        for item,seq in group:
            assert len(seq)==32768 and seq[-len(suffix):].tolist()==suffix and seq[pos]==old['answer_token_ids'][item['gold']]
            placements[item['item_id']]=dict(value_token_position=pos,target_block=target,sham_block=sham,query_start=start)
    out=R/'data/word-route-v0';out.mkdir(exist_ok=False);save(out/'placements.json',placements)
    seeds=[2026091660,2026091661];jobs=[];files={}
    order=[('baseline',seeds[0]),('baseline',seeds[1]),('target',seeds[0]),('sham',seeds[0]),('sham',seeds[1]),('target',seeds[1])]
    for condition,seed in order:
        j=dict(next(j for j in old['jobs'] if j['k']==32 and j['seed']==seed));j.pop('evaluation_k');j.pop('item_ids',None)
        ref=f'results/word-counterfactual-stage-v1/word-k32-seed{seed}-step128/result.json'
        f=R/'results/cloud-word-counterfactual-evidence-v1'/ref;files[ref]=f
        j.update(name=f'{condition}-seed{seed}',condition=condition,phase='replay' if condition=='baseline' else 'intervention',reference=ref,reference_sha256=sha(f));jobs.append(j)
    source_names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','word_route_intervention.py','check_word_route_intervention.py','eval_word_route_v0.py','run_word_route_stage_v0.py','prepare_word_route_v0.py','report_word_route_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in source_names}
    data={n:h for n,h in old['data_sha256'].items() if 'gentle32k' not in n and ('results/' not in n or n in files)}
    data['data/word-route-v0/placements.json']=sha(out/'placements.json')
    data['logs/word-route-preflight-v0-cloud.json']=sha(R/'logs/word-route-preflight-v0-cloud.json')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,seeds=seeds,source_sha256=sources,data_sha256=data,evaluation_gpu=old['evaluation_gpu'],answer_token_ids=old['answer_token_ids'],calibration_max_abs_error=1e-6,task_replay_max_abs_error=1e-6,maximum_seconds=2700,maximum_job_seconds=420,expected_task_predictions=384,optimizer_updates=0,maximum_gpu_cost_usd_excluding_setup_storage=2700/3600*.74,parent_audit_sha256=sha(R/'results/word-operator-swap-audit-v0/result.json'),bootstrap=dict(unit='background_family',n=32,draws=10000,seed=2026091697),primary_contrasts=['target-baseline','sham-baseline','target-sham'],scope='Privileged known-value-block diagnostic on reused development data, not a deployable method, independent validation, novelty or speed evidence. Freeze128-step K32 weights, bothseeds, all64longprompts. Bothbaseline replays must pass before any intervention. Only question-suffix queries at all24layers/all14heads change. Preserve local block and32unique causalblocks. Force value-containing block or predetermined target+2 sham block; discard one block deterministically in returned index order, NOT claimed to be lowest score. Does not force every preceding token of factual sentence. Original RACE oracle failures remain valid. Report allconditions andpaired family contrasts regardless of sign.')
    pp=R/'provenance/word-route-protocol-v0.json';assert not pp.exists();save(pp,p)
    for n in list(sources)+list(data):files.setdefault(n,R/n)
    files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/word-route-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
