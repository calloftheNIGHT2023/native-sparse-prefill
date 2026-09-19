"""Freeze the missing sparse64 cells of the matched 64/128 development curve."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 pp=R/'provenance/sparse-early64-protocol-v0.json';assert not pp.exists()
 parent=load(R/'provenance/lambada-natural-retry-protocol-v0.json');dense=load(R/'provenance/dense-early64-protocol-v0.json');jobs=[];refs=[]
 mirror=R/'results/cloud-matched-restore-training-evidence-v0'
 for seed in [2026091660,2026091661]:
  for restore in [False,True]:
   j=dict(next(x for x in parent['jobs'] if x.get('training_k')==32 and x['seed']==seed and x['restore_qk']==restore and x['step']==128))
   tr=load(mirror/j['training_result']);j.update(name=f'sparse64-seed{seed}-restore{int(restore)}',step=64,phase='same_step_control',path=j['path'].replace('checkpoint-128','checkpoint-64'))
   j['sha256']=sha(mirror/j['path']);j['calibration_values']=next(e['common_dense_calibration_values'] for e in tr['evaluations'] if e['step']==64)
   assert j['k']==0 and j['identity']['training_k']==32
   eventpath=(Path(j['training_result']).parent/'events.jsonl').as_posix();events=[json.loads(s) for s in (mirror/eventpath).read_text().splitlines()];e=next(x for x in events if x['event']=='checkpoint_reload_verified' and x['step']==64)
   j['early_process_seconds_utc']=(datetime.fromisoformat(e['utc'])-datetime.fromisoformat(tr['started_utc'])).total_seconds();j['early_step_seconds']=sum(x['seconds'] for x in tr['rows'][:64]);assert j['early_process_seconds_utc']>j['early_step_seconds']>0
   j['timing_events_path']=(mirror/eventpath).relative_to(R).as_posix();j['timing_events_sha256']=sha(mirror/eventpath);jobs.append(j)
  dj=next(x for x in dense['jobs'] if x['seed']==seed and not x['restore_qk']);rel='results/cloud-dense-early64-evidence-v0/results/dense-early64-stage-v0/'+dj['name']+'/result.json'
  refs.append(dict(seed=seed,kind='dense64',path=rel,sha256=sha(R/rel)))
 names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','prepare_sparse_early64_v0.py','eval_sparse_early64_v0.py','run_sparse_early64_stage_v0.py','report_sparse_early64_v0.py','collect_sparse_early64_v0.py'];sources={'scripts/'+n:sha(R/'scripts'/n) for n in names}
 old=load(R/'provenance/dense-early64-protocol-v0.json');data={n:sha(R/n) for n in old['data_sha256'] if n.startswith('data/')}
 data['provenance/dense-early64-protocol-v0.json']=sha(R/'provenance/dense-early64-protocol-v0.json')
 data['results/dense-early64-audit-v0/result.json']=sha(R/'results/dense-early64-audit-v0/result.json')
 for ref in refs:data[ref['path']]=ref['sha256']
 for j in jobs:data[j['timing_events_path']]=j['timing_events_sha256']
 primary=dict(comparison='sparse64 original/restored each versus dense64 original',accuracy_margin_pp=5,ppl_ratio_margin=1.05,bootstrap_seed=2026091708,draws=20000,quantiles=[.0125,.9875],method='Both fixed candidates; each intersection across five quality endpoints and cost in both seeds. Conditional97.5%two-sided CIs, conservative two-candidate Bonferroni. Average seeds within independent-unit proxy then paired bootstrap:32word-backgrounds,512LAMBADAitems,9Wikiwindows,16PGbooks. Accuracy lower>-5pp;PPLratio upper<=1.05. Exposed development only, not fresh confirmation.')
 p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,references=refs,source_sha256=sources,data_sha256=data,evaluation_gpu=parent['evaluation_gpu'],answer_token_ids=old['answer_token_ids'],calibration_max_abs_error=1e-6,expected_task_predictions=2580,expected_gradient_passes=0,expected_nll_forwards=244,optimizer_updates=0,maximum_seconds=1500,maximum_job_seconds=350,maximum_gpu_cost_usd_excluding_setup_storage=1500*.74/3600,primary=primary,scope='Complete missing sparse64 cells, no new training. Same4090 matched128-run midpoint checkpoints, replay common-dense calibration1e-6 before restoring QK. Both training arms use UTC from start to64 reload for cost, with sparse restoration/serialization added; require still cheaper with1second penalty. Includes reload diagnostic overhead, not pure deployment or invoice. No repeated forwards counted as independent evidence. QKRestore known baseline. No hyperparameter search, inference speed requirement, changed margins, or paper novelty claim. Dense64 early-control uncertainty does not prove dense inferior. This stage assesses sparse64 quality at the same64 update count;128 comparisons remain descriptive. Freeze independent confirmation and prior-art boundaries before new training.')
 save(pp,p);files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/sparse-early64-launch-v0.tar.gz';assert not a.exists();entries=[]
 with tarfile.open(a,'w:gz') as t:
  for n,f in sorted(files.items()):
   raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
