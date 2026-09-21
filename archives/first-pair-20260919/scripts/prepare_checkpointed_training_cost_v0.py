"""Freeze matched-memory dense/K32 training-body cost without weight updates."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    assert load(R/'results/word-suffix-profile-audit-v0/result.json')['status']=='verified'
    old=load(R/'provenance/word-suffix-cost-protocol-v0.json');job=dict(next(j for j in old['jobs'] if j['k']==0 and j['seed']==2026091660));job.update(name='matched-checkpointing-training-body',phase='training_cost')
    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','benchmark_checkpointed_training_v0.py','run_checkpointed_training_cost_stage_v0.py','prepare_checkpointed_training_cost_v0.py','report_checkpointed_training_cost_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in names};data={n:sha(R/n) for n in ['data/32k-expanded-training-v0/config.json','data/32k-expanded-training-v0/train-calibration.npz']}
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=[job],source_sha256=sources,data_sha256=data,evaluation_gpu=old['evaluation_gpu'],calibration_max_abs_error=1e-6,orders=[[0,32],[32,0],[0,32]],windows=[0,1,2],expected_task_predictions=0,expected_gradient_passes=18,optimizer_updates=0,maximum_seconds=660,maximum_job_seconds=600,maximum_gpu_cost_usd_excluding_setup_storage=660*.74/3600,parent_audit_sha256=sha(R/'results/word-suffix-profile-audit-v0/result.json'),scope='One fixed dense-trained128-step checkpoint. Native4window NLL replay1e-6. Both dense/K32 use same nonreentrant gradientcheckpointing, FP32parameters/BF16autocast, frozenhead chunk256backprop and clipping. Four2048token forward/backward checks compare checkpointing on/off, original loss.001 and gradientrelativeL2.02 gates. Two32Kwarmups then3windows x2modes x2repeats=12timed fullforward/backwards,18totalgradientpasses. Repeats require allclosegrad atol1e-6 rtol1e-4 andlossgap<.001.0optimizerupdates,weightsdigestunchanged. Timing excludes optimizer, inputtransfer, gradCPUverification and data/model loading. Report trainingbodycost, notformaltrain-step speed or convergence. Original6000Ada recipe used39-40GiB; current4090usesdifferent matchedmemory-saving settings andmustnot mixtimings. Allfailed passes preserved; do not relax numericalgates.')
    pp=R/'provenance/checkpointed-training-cost-protocol-v0.json';assert not pp.exists();save(pp,p);files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/checkpointed-training-cost-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
