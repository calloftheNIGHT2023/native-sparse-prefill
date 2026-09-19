"""Freeze a known posthoc baseline on the now-exposed second word/background set."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, io, json, tarfile
R = Path(__file__).resolve().parents[1]
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main():
    pp=R/'provenance/qk-restore-transfer-protocol-v0.json'; assert not pp.exists()
    # Preserve frozen parents. New files contain exactly the common-dense evaluator
    # plus the previously audited QK intervention, before any task predictions.
    old=(R/'scripts/eval_fresh_word_dense_v0.py').read_text(encoding='utf-8')
    new=old.replace('provenance/fresh-word-dense-protocol-v0.json','provenance/qk-restore-transfer-protocol-v0.json')
    restore=(R/'scripts/eval_qk_restore_baseline_v0.py').read_text(encoding='utf-8')
    block=restore[restore.index('  before={n:p.detach()'):restore.index('  report=np.load')]
    anchor="  event('common_dense_evaluation',training_k=a.k,evaluation_k=0)\n"
    assert new.count(anchor)==1
    new=new.replace(anchor,anchor+block)
    new=new.replace("result=dict(status='complete',job=job", "result=dict(status='complete',ablation=ablation,job=job")
    path=R/'scripts/eval_qk_restore_transfer_v0.py'; assert not path.exists();path.write_text(new,encoding='utf-8')
    ctrl=(R/'scripts/run_fresh_word_dense_stage_v0.py').read_text(encoding='utf-8').replace('fresh-word-dense','qk-restore-transfer').replace('fresh_word_dense','qk_restore_transfer')
    ctrl=ctrl.replace("assert count in [5,133,p['expected_task_predictions']]", "assert count==p['expected_task_predictions']")
    path=R/'scripts/run_qk_restore_transfer_stage_v0.py';assert not path.exists();path.write_text(ctrl,encoding='utf-8')
    parent=load(R/'provenance/fresh-word-dense-protocol-v0.json')
    jobs=[]; refs=[]
    for j in parent['jobs']:
        if j['step']!=128: continue
        job=dict(j);job.update(name=j['name'].replace('common-dense','qk-restored'),phase='development_transfer');jobs.append(job)
        ref='results/cloud-fresh-word-dense-evidence-v0/results/fresh-word-dense-stage-v0/'+j['name']+'/result.json'
        refs.append(dict(k=j['k'],seed=j['seed'],path=ref,sha256=sha(R/ref)))
    assert len(jobs)==4
    sources={n:h for n,h in parent['source_sha256'].items() if n.split('/')[-1] in ['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py']}
    for n in ['prepare_qk_restore_transfer_v0.py','eval_qk_restore_transfer_v0.py','run_qk_restore_transfer_stage_v0.py','report_qk_restore_transfer_v0.py','collect_qk_restore_transfer_v0.py']:
        sources['scripts/'+n]=sha(R/'scripts'/n)
    data=dict(parent['data_sha256'])
    for n in ['provenance/fresh-word-dense-protocol-v0.json','provenance/qk-restore-baseline-protocol-v0.json','docs/qk-restore-overlap-2026-09-16.md']:
        data[n]=sha(R/n)
    for ref in refs: data[ref['path']]=ref['sha256']
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,references=refs,source_sha256=sources,data_sha256=data,evaluation_gpu=parent['evaluation_gpu'],answer_token_ids=parent['answer_token_ids'],calibration_max_abs_error=1e-6,base_long_min_correct=96,expected_task_predictions=532,expected_gradient_passes=0,optimizer_updates=0,maximum_seconds=1200,maximum_job_seconds=290,maximum_gpu_cost_usd_excluding_setup_storage=1200*.74/3600,
        primary=dict(endpoint=128,evaluation_k=0,bootstrap_seed=2026091701,bootstrap_draws=20000,noninferiority_margin_pp=5,primary_contrast='restored sparse versus original dense',secondary_contrasts=['restored sparse versus restored dense','restoration effect within each training mode','difference of restoration effects']),
        scope='Development transfer screen of existing QK-Restore, not a new method. The second 32-background/four-word task set has already been exposed; not independent confirmation. Four unchanged endpoint128 QKVO LoRA checkpoints, two seeds, same common-dense evaluation. After native calibration replay <=1e-6, zero exactly48 Q/K LoRA B tensors;144 other adapter tensors unchanged. Reuse hash-verified original predictions, do not repeat base ability evaluation. Primary paired32-background CI averages both seeds and allfour facts; conditional on seeds/template. Screen passes only if allshort prompts correct and lower95% accuracy difference bound >-5pp against original dense. Report symmetric restored dense comparison as well. No training, new NLL, inference-speed or novelty claim. If failed, do not launch more training under same recipe. If passed, new external natural-quality validation and matched actual training cost remain necessary.')
    save(pp,p)
    files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp
    a=R/'exports/qk-restore-transfer-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof)
    print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
