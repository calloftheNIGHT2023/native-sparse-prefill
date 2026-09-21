"""Freeze interleaved, cache-producing whole-model prefill timing screen."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    assert load(R/'results/word-suffix-audit-v0/result.json')['status']=='verified'
    old=load(R/'provenance/word-suffix-protocol-v0.json');files={};refs=[]
    for ref in old['references']:
        v=dict(ref);v['condition']='DD' if v.pop('evaluation_k')==0 else 'SS';refs.append(v);files[v['path']]=R/'results/cloud-word-suffix-evidence-v0'/v['path']
    for j in old['jobs']:
        n='results/word-suffix-stage-v0/'+j['name']+'/result.json';f=R/'results/cloud-word-suffix-evidence-v0'/n;files[n]=f;refs.append(dict(path=n,sha256=sha(f),training_k=j['k'],seed=j['seed'],condition='SD'))
    ids=[f'word-cf-{i:03}-cf{cf}' for i in [0,9,18,27] for cf in [0,1]]
    modes=['DD','SS','SD'];schedule=[dict(item_id=ids[0],mode=m,phase='warmup',round=-1,order=i) for i,m in enumerate(modes)]
    for rep in range(3):
        for index,item in enumerate(ids):
            shift=(rep+index)%3;order=modes[shift:]+modes[:shift]
            for m in order:schedule.append(dict(item_id=item,mode=m,phase='timed',round=rep,order=len(schedule)))
    assert len(schedule)==75
    jobs=[]
    for k,seed in [(0,2026091660),(32,2026091660),(32,2026091661),(0,2026091661)]:
        j=dict(next(x for x in old['jobs'] if x['k']==k and x['seed']==seed));j.update(name=f'prefill-cost-k{k}-seed{seed}',phase='prefill_cost');jobs.append(j)
    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','word_suffix_attention.py','bench_word_suffix_v0.py','run_word_suffix_cost_stage_v0.py','prepare_word_suffix_cost_v0.py','report_word_suffix_cost_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in names};data={n:h for n,h in old['data_sha256'].items() if not n.startswith('results/')}
    for n,f in files.items():data[n]=sha(f)
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,references=refs,item_ids=ids,schedule=schedule,source_sha256=sources,data_sha256=data,suffix_token_ids=old['suffix_token_ids'],answer_token_ids=old['answer_token_ids'],evaluation_gpu=old['evaluation_gpu'],calibration_max_abs_error=1e-6,task_replay_max_abs_error=1e-6,maximum_seconds=1500,maximum_job_seconds=330,expected_task_predictions=300,optimizer_updates=0,maximum_gpu_cost_usd_excluding_setup_storage=1500/3600*.74,parent_audit_sha256=sha(R/'results/word-suffix-audit-v0/result.json'),cost_screen=dict(minimum_median_round_saving_percent=5,require_all_rounds_positive=True),scope='Whole-model cache-producing prefill through final-token LM-head logits on one RTX4090, including MoBA routing and dense suffix recomputation. GPU-resident tokeninputs, batch1, 32K, use_cache=True. Excludes tokenization, transfer, queueing, network and checkpoint startup; not serving TTFT or training cost. 4frozen weights x (3warmup+3rounds x8prompts x3modes)=300. Eachlogit vector must replay existing no-cache reference within1e-6; fail closed. Fixedmode Latinrotation per prompt balances ordering over three rounds. Allwarmup records retained but excluded from timings. Fourpositions/twofacts per position selected by IDs beforetiming, notaccuracy. Report everymodel/round; repeated runs notindependent scientific samples. Existing quality gaps remain; timing does not prove quality equivalence or novelty. At least5percent median andpositive savings in all12 model-rounds is a resource-allocation screen only.')
    pp=R/'provenance/word-suffix-cost-protocol-v0.json';assert not pp.exists();save(pp,p)
    for n in list(sources)+list(data):files.setdefault(n,R/n)
    files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/word-suffix-cost-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
