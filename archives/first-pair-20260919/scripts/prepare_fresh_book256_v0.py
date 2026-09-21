"""Freeze nine existing-checkpoint evaluations on the new256-book background set."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 pp=R/'provenance/fresh-book256-protocol-v0.json';assert not pp.exists();manifest=load(R/'data/fresh-book256-v0/manifest.json');assert manifest['builder_sha256']==sha(R/'scripts/build_fresh_book256_v0.py') and len(manifest['families'])==256
 parent=load(R/'provenance/lambada-natural-retry-protocol-v0.json');jobs=[dict(parent['jobs'][0])];mirror=R/'results/cloud-matched-restore-training-evidence-v0'
 for step in [64,128]:
  for k,seed in [(0,2026091660),(32,2026091660),(32,2026091661),(0,2026091661)]:
   j=dict(next(x for x in parent['jobs'] if x.get('training_k')==k and x['seed']==seed and x['restore_qk']==(k==32) and x['step']==128));j.update(name=f'train{k}-step{step}-seed{seed}',phase='fresh_background_confirmation',step=step,path=j['path'].replace('checkpoint-128',f'checkpoint-{step}'))
   tr=load(mirror/j['training_result']);j['sha256']=sha(mirror/j['path']);j['calibration_values']=next(x['common_dense_calibration_values'] for x in tr['evaluations'] if x['step']==step);jobs.append(j)
 names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','build_fresh_book256_v0.py','prepare_fresh_book256_v0.py','eval_fresh_book256_v0.py','run_fresh_book256_stage_v0.py','report_fresh_book256_v0.py','collect_fresh_book256_v0.py'];sources={'scripts/'+n:sha(R/'scripts'/n) for n in names}
 files=['provenance/fresh-book256-data-audit-v0.json','provenance/fresh-book256-precision-planning-v0.json','data/32k-expanded-training-v0/config.json','data/32k-expanded-training-v0/train-calibration.npz','data/fresh-book256-v0/tasks.npz','data/fresh-book256-v0/tasks.json','data/fresh-book256-v0/manifest.json','data/fresh-book256-v0/design-before-data.json','data/fresh-book256-v0/official-train-listing-first1000.json','results/dense-early64-audit-v0/result.json','results/sparse-early64-audit-v0/result.json','results/matched-restore-training-audit-v0/result.json','docs/sparse-training-overlap-addendum-2026-09-17.md'];data={n:sha(R/n) for n in files}
 def pairs(ka,sa,kb,sb):return [[f'train{ka}-step{sa}-seed{s}',f'train{kb}-step{sb}-seed{s}'] for s in [2026091660,2026091661]]
 primary=dict(bootstrap_seed=2026091710,draws=20000,quantiles=[.05/6,1-.05/6],accuracy_margin_pp=5,comparisons=[dict(name='sparse64restored_minus_dense64',pairs=pairs(32,64,0,64)),dict(name='sparse128restored_minus_dense128',pairs=pairs(32,128,0,128)),dict(name='dense64_minus_sparse128restored',pairs=pairs(0,64,32,128))],method='Average the two old training seeds within each book, then jointly resample256books. Four fact values per book are clustered. Three fixed98.333%two-sided paired CIs, conservative Bonferroni. Noninferior only lower>-5pp. Same oldtemplate/words, new source books; this does not validate new tasks or seeds.')
 p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_sha256=sources,data_sha256=data,evaluation_gpu=parent['evaluation_gpu'],answer_token_ids=manifest['label_token_ids'],calibration_max_abs_error=1e-6,expected_task_predictions=9261,expected_gradient_passes=0,expected_nll_forwards=36,optimizer_updates=0,maximum_seconds=10800,maximum_job_seconds=1150,maximum_gpu_cost_usd_excluding_setup_storage=10800*.74/3600,base_gate=dict(short_correct=4,long_correct_min=768),primary=primary,scope=manifest['scope']+' Same frozen checkpoint0/64/128, common-dense calibration before intervention. Four dense and four sparse-restored endpoints plusbase; no new training. No optimizer or GPU-cost claims inferred from evaluation wall. Overlap screening uses128token chunks at stride64 only, not exhaustive near-duplicate detection. Distinct book IDs are clustering proxies; different editions/authors can still correlate.  Natural-task and cost evidence remain older audited data, not new independent outcomes. Any base gate failure stops after1029 predictions. No model/threshold selection after outcomes; one bounded evaluation stage, no indefinite training.')
 save(pp,p);allfiles={n:R/n for n in list(sources)+list(data)};allfiles[pp.relative_to(R).as_posix()]=pp;a=R/'exports/fresh-book256-launch-v0.tar.gz';assert not a.exists();entries=[]
 with tarfile.open(a,'w:gz') as t:
  for n,f in sorted(allfiles.items()):
   raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
