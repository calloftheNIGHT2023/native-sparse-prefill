"""Freeze dense64 (plain/restored) against existing sparse128 restored endpoints."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io,ast
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    pp=R/'provenance/dense-early64-protocol-v0.json';assert not pp.exists()
    code=(R/'scripts/eval_lambada_natural_retry_v0.py').read_text(encoding='utf-8').replace('provenance/lambada-natural-retry-protocol-v0.json','provenance/dense-early64-protocol-v0.json')
    anchor="  if job['restore_qk']:\n";assert code.count(anchor)==1;code=code.replace(anchor,"  torch.cuda.synchronize();restore_start=time.perf_counter()\n"+anchor)
    anchor="  expected_digest=hashlib.sha256"
    addition="  if job['restore_qk']:\n   torch.save(dict(params={n:p.detach().cpu() for n,p in params.items()},step=64,intervention='known_QK_Restore'),out/'restored-adapter.pt')\n  torch.cuda.synchronize();restore_seconds=time.perf_counter()-restore_start if job['restore_qk'] else 0.\n"
    assert code.count(anchor)==1;code=code.replace(anchor,addition+anchor)
    anchor="  meta=json.loads((R/'data/lambada-natural-v0/tasks.json').read_text())"
    extra=(R/'scripts/dense_early64_extra_eval_v0.txt').read_text(encoding='utf-8');assert code.count(anchor)==1;code=code.replace(anchor,extra+anchor)
    code=code.replace("result=dict(status='complete',job=job", "result=dict(status='complete',wiki_nll=wiki,pg19=pgrows,word_predictions=word_preds,restore_seconds=restore_seconds,nll_forwards=61,job=job")
    code=code.replace('task_predictions=len(predictions),calibration_nll_forwards=4','task_predictions=len(predictions)+len(word_preds),calibration_nll_forwards=4')
    ast.parse(code);path=R/'scripts/eval_dense_early64_v0.py';assert not path.exists();path.write_text(code,encoding='utf-8')
    ctrl=(R/'scripts/run_lambada_natural_retry_stage_v0.py').read_text(encoding='utf-8').replace('lambada-natural-retry','dense-early64').replace('lambada_natural_retry','dense_early64').replace("count in [1024,p['expected_task_predictions']]","count==p['expected_task_predictions']")
    path=R/'scripts/run_dense_early64_stage_v0.py';assert not path.exists();ast.parse(ctrl);path.write_text(ctrl,encoding='utf-8')
    parent=load(R/'provenance/lambada-natural-retry-protocol-v0.json');jobs=[];refs=[]
    for seed in [2026091660,2026091661]:
        for restore in [False,True]:
            j=dict(next(x for x in parent['jobs'] if x.get('training_k')==0 and x['seed']==seed and x['restore_qk']==restore and x['step']==128))
            mirror=R/'results/cloud-matched-restore-training-evidence-v0';tr=load(mirror/j['training_result']);j.update(name=f'dense64-seed{seed}-restore{int(restore)}',step=64,phase='early_stop_control',path=j['path'].replace('checkpoint-128','checkpoint-64'))
            j['sha256']=sha(mirror/j['path']);j['calibration_values']=next(e['common_dense_calibration_values'] for e in tr['evaluations'] if e['step']==64)
            eventpath=str(Path(j['training_result']).parent/'events.jsonl').replace('\\','/');events=[json.loads(s) for s in (mirror/eventpath).read_text().splitlines()];event=next(e for e in events if e['event']=='checkpoint_reload_verified' and e['step']==64)
            j['early_process_seconds_utc']=(datetime.fromisoformat(event['utc'])-datetime.fromisoformat(tr['started_utc'])).total_seconds();j['early_step_seconds']=sum(x['seconds'] for x in tr['rows'][:64]);assert j['early_process_seconds_utc']>j['early_step_seconds']>0
            j['timing_events_path']=str((mirror/eventpath).relative_to(R)).replace('\\','/');j['timing_events_sha256']=sha(mirror/eventpath);jobs.append(j)
        sj=next(x for x in parent['jobs'] if x.get('training_k')==32 and x['seed']==seed and x['restore_qk'])
        for kind,rel in [('matched','results/cloud-matched-restore-training-evidence-v0/'+sj['training_result']),('lambada','results/cloud-lambada-natural-retry-evidence-v0/results/lambada-natural-retry-stage-v0/'+sj['name']+'/result.json')]:refs.append(dict(seed=seed,kind=kind,path=rel,sha256=sha(R/rel)))
    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','prepare_dense_early64_v0.py','dense_early64_extra_eval_v0.txt','eval_dense_early64_v0.py','run_dense_early64_stage_v0.py','report_dense_early64_v0.py','collect_dense_early64_v0.py'];sources={'scripts/'+n:sha(R/'scripts'/n) for n in names}
    data={n:sha(R/n) for n in ['data/32k-expanded-training-v0/config.json','data/32k-expanded-training-v0/train-calibration.npz','data/32k-expanded-training-v0/report.npz','data/pg19-external-v0/windows.npz','data/pg19-external-v0/manifest.json','data/fresh-word-dense-v0/tasks.npz','data/fresh-word-dense-v0/tasks.json','data/fresh-word-dense-v0/manifest.json','data/lambada-natural-v0/tasks.npz','data/lambada-natural-v0/tasks.json','data/lambada-natural-v0/manifest.json','docs/next-critical-control-2026-09-17.md']}
    for ref in refs:data[ref['path']]=ref['sha256']
    for j in jobs:data[j['timing_events_path']]=j['timing_events_sha256']
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,references=refs,source_sha256=sources,data_sha256=data,evaluation_gpu=parent['evaluation_gpu'],answer_token_ids=load(R/'data/fresh-word-dense-v0/manifest.json')['label_token_ids'],calibration_max_abs_error=1e-6,expected_task_predictions=2580,expected_gradient_passes=0,expected_nll_forwards=244,optimizer_updates=0,maximum_seconds=1500,maximum_job_seconds=350,maximum_gpu_cost_usd_excluding_setup_storage=1500*.74/3600,
      primary=dict(comparison='dense64 each of original/restored versus sparse128 restored',accuracy_margin_pp=5,ppl_ratio_margin=1.05,bootstrap_seed=2026091707,draws=20000,quantiles=[.0125,.9875],method='Two fixed early-dense candidates. For each, intersection of both accuracies lower>-5pp, Wiki/fullPG/tailPG PPL ratio upper<=1.05, bothseeds cheaper. Use97.5%two-sided intervals per candidate as conservative Bonferroni over two candidates; average seeds then paired resampling within32word-backgrounds,512LAMBADAitems,9Wikiwindows,16PGbooks. Exposed development control, conditional uncertainty only.'),
      scope='No new training. Four matched4090 dense64 checkpoint evaluations: both seeds plain and knownQKRestored. Compare against immutable sparse128restored references from matched training and LAMBADA. Same dense attention, native common-cal replay1e-6, full-vocabulary natural task. Do not choose64 from a sweep; it is the already saved midpoint control. Cost is archived UTC delta to64 checkpoint reload (conservatively includes reload not needed for early stopping) plus newly timed restore/adapter serialization when applicable. This differs from monotonic endpoint128 runtime; report both sources and 1second clock/overhead sensitivity, no false exact timing claim. Five quality endpoints jointly, include symmetric restoration, no task/margin selection. All eval data now exposed, not fresh generalization. Passing means a cheaper early dense candidate satisfies these quality tolerances, undermining a broad same-quality sparse-cost claim while preserving fixed128-step saving. If no candidate passes, that alone does not prove optimal sparse time-to-quality. Known QKRestore not new.')
    save(pp,p);files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/dense-early64-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
