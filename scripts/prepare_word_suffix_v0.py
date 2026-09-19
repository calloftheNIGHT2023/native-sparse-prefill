"""Freeze answer-location-blind dense question suffix on fixed K32 prefixes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile,io
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    pre=load(R/'logs/word-suffix-preflight-v1-cloud.json');assert pre['status']=='passed' and pre['helper_sha256']==sha(R/'scripts/word_suffix_attention.py')
    assert load(R/'results/word-route-audit-v0/result.json')['status']=='verified'
    old=load(R/'provenance/word-operator-swap-protocol-v0.json');jobs=[];files={};refs=[]
    for oldjob in old['jobs']:
        j=dict(oldjob);j.pop('evaluation_k');j.pop('item_ids',None);j.update(name=f"suffix-train{j['k']}-seed{j['seed']}",phase='dense_suffix');jobs.append(j)
    for native in old['native_references']:
        n=native['path'];f=R/'results/cloud-word-counterfactual-evidence-v1'/n;files[n]=f;refs.append(dict(path=n,sha256=sha(f),training_k=native['k'],evaluation_k=native['k'],seed=native['seed']))
    for j in old['jobs']:
        n='results/word-operator-swap-stage-v0/'+j['name']+'/result.json';f=R/'results/cloud-word-operator-swap-evidence-v0'/n;files[n]=f;refs.append(dict(path=n,sha256=sha(f),training_k=j['k'],evaluation_k=j['evaluation_k'],seed=j['seed']))
    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','word_suffix_attention.py','check_word_suffix_v1.py','eval_word_suffix_v0.py','run_word_suffix_stage_v0.py','prepare_word_suffix_v0.py','report_word_suffix_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in names};data={n:h for n,h in old['data_sha256'].items() if 'gentle32k' not in n and not n.startswith('results/')}
    for n,f in files.items():data[n]=sha(f)
    for n in ['logs/word-suffix-preflight-v1-cloud.json','logs/word-suffix-preflight-v0-failed.log','docs/word-route-overlap-check-2026-09-16.md']:data[n]=sha(R/n)
    tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True);suffix=tok.encode('\n\nQuestion: What is the secret word?\nAnswer:',add_special_tokens=False)
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,seeds=[2026091660,2026091661],source_sha256=sources,data_sha256=data,references=refs,suffix_token_ids=suffix,evaluation_gpu=old['evaluation_gpu'],answer_token_ids=old['answer_token_ids'],calibration_max_abs_error=1e-6,maximum_seconds=1200,maximum_job_seconds=270,expected_task_predictions=256,optimizer_updates=0,maximum_gpu_cost_usd_excluding_setup_storage=1200/3600*.74,parent_audit_sha256=sha(R/'results/word-route-audit-v0/result.json'),bootstrap=dict(unit='background_family',n=32,draws=10000,seed=2026091699),scope='Known dense-suffix baseline and phase diagnosis, NOT novel. Bothdense/K32 trained128-step weights andbothseeds. Native calibration unchanged before intervention. Every prefix query uses K32; only the complete fixedquestion suffix uses full causal attention. The operator receives sequence positions but no gold labels, valuepositions or targetblock. Reuses allnative andswapped SS/DD references from same4090. No training, no independent validation or end-to-end acceleration claim. Diagnostic computes anddiscards sparse suffix output before dense recomputation; any eventual speed test must include this overhead. Preserve failedv0 numerical preflight due missing CUBLAS deterministic env; v1 fixes environment only.')
    pp=R/'provenance/word-suffix-protocol-v0.json';assert not pp.exists();save(pp,p)
    for n in list(sources)+list(data):files.setdefault(n,R/n)
    files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/word-suffix-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof,suffix_tokens=len(suffix))))
if __name__=='__main__':main()
