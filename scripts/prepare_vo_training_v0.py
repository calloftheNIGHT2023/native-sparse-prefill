"""Freeze a matched V/O-only continued-training development experiment."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    assert load(R/'results/qk-restore-baseline-audit-v0/result.json')['status']=='verified'
    parent=load(R/'provenance/common-dense-quality-protocol-v0.json');jobs=[]
    for seed,order in [(2026091660,[0,32]),(2026091661,[32,0])]:
        template=dict(next(j for j in parent['jobs'] if j['k']==0 and j['seed']==seed));n=f'results/expanded76-stage-v0/k0-seed{seed}/checkpoint-0.pt'
        for k in order:
            j=dict(template);j.update(name=f'vo-train{k}-seed{seed}',phase='vo_training',step=0,path=n,sha256=sha(R/'results/cloud-expanded76-evidence-v0'/n),training_k=k);jobs.append(j)
    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','train_vo_only_v0.py','run_vo_training_stage_v0.py','prepare_vo_training_v0.py','report_vo_training_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in names}
    data={n:sha(R/n) for n in ['data/32k-expanded-training-v0/config.json','data/32k-expanded-training-v0/train-calibration.npz','data/32k-expanded-training-v0/report.npz','data/word-counterfactual-v0/tasks.npz','data/word-counterfactual-v0/tasks.json','docs/vo-training-overlap-2026-09-17.md']}
    labels=load(R/'data/word-counterfactual-v0/manifest.json')['label_token_ids']
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_sha256=sources,data_sha256=data,evaluation_gpu=parent['evaluation_gpu'],answer_token_ids=labels,calibration_max_abs_error=1e-6,steps_per_job=128,expected_optimizer_updates=512,expected_task_predictions=768,expected_gradient_passes=8,maximum_seconds=3900,maximum_job_seconds=930,maximum_gpu_cost_usd_excluding_setup_storage=3900*.74/3600,parent_audit_sha256=sha(R/'results/qk-restore-baseline-audit-v0/result.json'),screen=dict(endpoint=128,accuracy_noninferiority_margin_pp=5,perplexity_ratio_upper_margin=1.05,median_complete_step_saving_min_percent=5,positive_step_saving_each_seed=True),scope='Development experiment, not independent confirmation or new method. Same pre-training checkpoint within each seed; initialize original192 LoRA tensors, freeze96 Q/K tensors, train only96 V/O tensors (540672parameters). Dense versus all128updates K32, same76windows/order/LR .001 warmup4 original64step schedule then .0001 floor, AdamW, clipping, nonreentrant checkpointing and4090. Two full32K gradient repeats per job before training, existing tolerance atol1e-6 rtol1e-4/loss.001, zero updates; native dense0 calibration replay1e-6. Save0/64/128 checkpoints; reload64 and assert parameter digest/LR unchanged. Common-dense calibration64/128 and final192 old word-task predictions/9oldreport NLL. Primary128 endpoint only. Screen quality jointly and cost from actual training; not infer equivalence from non-significance. Two seeds, old exposed data; all-sparse applies only to this LoRA continued-training phase, model was previously dense-pretrained. No inference speed objective. Distinguish step sum, loop wall including mid-calibration/checkpoints, training-completed process time including setup/preflight, and whole job including final evaluation.')
    pp=R/'provenance/vo-training-protocol-v0.json';assert not pp.exists();save(pp,p);files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/vo-training-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
