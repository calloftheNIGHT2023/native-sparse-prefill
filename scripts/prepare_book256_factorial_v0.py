"""Freeze missing cells of an exposed-data factorial diagnostic, not a new method."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, tarfile, io, copy
R=Path(__file__).resolve().parents[1]
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 pp=R/'provenance/book256-factorial-protocol-v0.json';assert not pp.exists()
 old=load(R/'provenance/fresh-book256-protocol-v0.json');prior=load(R/'results/fresh-book256-resume-audit-v0/result.json');assert prior['status']=='verified'
 jobs=[]
 for step in [64,128]:
  for k,seed in [(0,2026091660),(32,2026091660),(32,2026091661),(0,2026091661)]:
   j=copy.deepcopy(next(x for x in old['jobs'] if x['step']==step and x['training_k']==k and x['seed']==seed))
   j.update(name=j['name']+'-'+('restore' if k==0 else 'original'),restore_qk=k==0,phase='exposed_factorial_diagnostic');jobs.append(j)
 names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','eval_book256_factorial_v0.py','run_book256_factorial_stage_v0.py','report_book256_factorial_v0.py','collect_book256_factorial_v0.py','prepare_book256_factorial_v0.py']
 sources={'scripts/'+n:sha(R/'scripts'/n) for n in names}
 files=list(old['data_sha256'])+['provenance/fresh-book256-protocol-v0.json','provenance/fresh-book256-resume-protocol-v0.json','results/fresh-book256-resume-audit-v0/result.json','docs/qk-restore-overlap-2026-09-16.md','docs/book256-factorial-plan-2026-09-17.md']
 data={n:sha(R/n) for n in files}
 comparisons=[]
 for step in [64,128]:
  pairterms=[];interterms=[]
  for seed in [2026091660,2026091661]:
   d=f'train0-step{step}-seed{seed}';s=f'train32-step{step}-seed{seed}'
   pairterms.append([[s,1],[d+'-restore',-1]])
   interterms.append([[s,1],[s+'-original',-1],[d+'-restore',-1],[d,1]])
  comparisons += [dict(name=f'symmetric_restored_step{step}',kind='noninferiority',seed_terms=pairterms),dict(name=f'restore_interaction_step{step}',kind='interaction',seed_terms=interterms)]
 p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_sha256=sources,data_sha256=data,evaluation_gpu=old['evaluation_gpu'],answer_token_ids=old['answer_token_ids'],calibration_max_abs_error=1e-6,expected_task_predictions=8232,expected_gradient_passes=0,expected_nll_forwards=32,optimizer_updates=0,maximum_seconds=7200,maximum_job_seconds=1150,maximum_gpu_cost_usd_excluding_setup_storage=1.48,hourly_gpu_usd_estimate=.74,hourly_rate_verified=False,parent_archives=[dict(path='exports/fresh-book256-user-stop-v0.tar.gz',sha256='f670c0f8a0b9db131b91384f235f9f9ce7f175efa0011bd9bdd79e5473e1afdc'),dict(path='exports/fresh-book256-resume-evidence-v0.tar.gz',sha256=prior['archive']['sha256'])],primary=dict(bootstrap_seed=2026091711,draws=20000,quantiles=[.05/8,1-.05/8],accuracy_margin_pp=5,comparisons=comparisons),scope='Adaptive follow-up chosen after original book256 outcomes. Same exposed256books, oldtemplate/words, two old training seeds. Complete missing dense-restored and sparse-original cells at64/128 without repeating existing predictions. Fixed4contrasts with98.75%conditional paired book-cluster intervals; symmetric noninferiority only lower>-5pp; interaction reported without equivalence claim. Not independent confirmation, naturalQA, new model or new training seed. Near-ceiling base and sparse-restored accuracy limits discrimination. GenericQK-Restore and sparse-finetuning/dense-eval are known methods; no originality claim. No quality selection by changing checkpoints, words or margins. No training or new cost benchmark: evaluation wall cannot establish training cost savings. Preserve original3contrasts, including cheaper dense64 passing5pp noninferiority against sparse128. Current GPU rate is an estimate, excludes setup idle storage and failed environment repair; not invoice.')
 save(pp,p);a=R/'exports/book256-factorial-launch-v0.tar.gz';assert not a.exists();entries=[]
 with tarfile.open(a,'w:gz') as t:
  for n in sorted(set(sources)|set(data)|{pp.relative_to(R).as_posix()}):
   f=R/n;raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
