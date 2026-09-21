"""Freeze matched-memory dense/K32 training-body cost without weight updates."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    assert load(R/'results/checkpointed-training-cost-audit-v0/result.json')['status']=='verified'
    old=load(R/'provenance/midpoint64-protocol-v2.json');jobs=[]
    for seed,order in [(2026091660,[0,32]),(2026091661,[32,0])]:
        for k in order:
            job=dict(next(j for j in old['jobs'] if j['k']==0 and j['seed']==seed and j['step']==128));job.update(name=f'cost-s{seed}-train{k}',training_k=k,phase='full_step_cost');jobs.append(job)

    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','benchmark_full_step_cost_v0.py','run_full_step_cost_stage_v0.py','prepare_full_step_cost_v0.py','report_full_step_cost_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in names};data={n:sha(R/n) for n in ['data/32k-expanded-training-v0/config.json','data/32k-expanded-training-v0/train-calibration.npz']}
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_sha256=sources,data_sha256=data,evaluation_gpu=old['evaluation_gpu'],calibration_max_abs_error=1e-6,steps_per_job=16,expected_optimizer_updates=64,scientific_updates=0,diagnostic_updates=64,expected_task_predictions=0,expected_gradient_passes=0,maximum_seconds=1200,maximum_job_seconds=280,maximum_gpu_cost_usd_excluding_setup_storage=1200*.74/3600,numerical_preflight_audit_sha256=sha(R/'results/checkpointed-training-cost-audit-v0/result.json'),scope='Cost-only diagnostic. Two seeds, each branches same dense-trained128-step weights and saved AdamW/scheduler state to16 dense or K32 updates, opposite order across seeds. Same nonreentrant checkpointing FP32parameters/BF16autocast. Primary all16 synchronized step sum includes CPU input creation/transfer, zero-grad, forward, chunked head backward, backbone backward/recompute, clipping, AdamW and scheduler. Secondary loop wall includes logs; whole-process wall includes model/data load, replay and checkpoint save. No fresh-quality, all-sparse training or convergence-cost claim. First2steps retained in primary total; steady14median only descriptive. Outputs and failed updates preserved. Numerical preflight is existing matched checkpointing loss/gradient/repeat audit; native calibration replay1e-6 per job before changes. The same source checkpoint is used within each seed to avoid timing comparisons between different weight states.')

    pp=R/'provenance/full-step-cost-protocol-v0.json';assert not pp.exists();save(pp,p);files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/full-step-cost-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
