"""Freeze common-attention quality diagnosis; no new training or claim of fresh testing."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    old=load(R/'provenance/midpoint64-protocol-v2.json');jobs=[]
    for j in old['jobs']:
        if j['k'] not in [0,32] or (j['step']==0 and j['k']!=0):continue
        j=dict(j);j.update(name=f"dense-eval-k{j['k']}-s{j['seed']}-t{j['step']}",phase='common_dense_quality');jobs.append(j)
    assert len(jobs)==9
    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','eval_common_dense_quality_v0.py','run_common_dense_quality_stage_v0.py','prepare_common_dense_quality_v0.py','report_common_dense_quality_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in names}
    data={n:sha(R/n) for n in ['data/32k-expanded-training-v0/config.json','data/32k-expanded-training-v0/train-calibration.npz','data/32k-expanded-training-v0/report.npz']}
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_sha256=sources,data_sha256=data,evaluation_gpu=old['evaluation_gpu'],calibration_max_abs_error=1e-6,expected_task_predictions=0,expected_gradient_passes=0,expected_nll_forwards=279,optimizer_updates=0,maximum_seconds=1800,maximum_job_seconds=240,maximum_gpu_cost_usd_excluding_setup_storage=1800*.74/3600,scope='Nine existing checkpoints: shared dense base, dense/K32 at64/128 steps and two seeds. All score the same nine previously exposed WikiText report windows using dense attention. Each has four native-attention calibration replay passes and27 dense NLL passes (9 full32K,9 full32K last4K-target,9 short4K). Primary descriptive endpoint is128-step paired full-window NLL delta and exp(mean delta) perplexity ratio. Secondary context benefit and64-step trajectory. No checkpoint selection, no fresh test or population equivalence claim; repeated corpus windows are not independent articles. Bootstrap intervals only conditional on these windows, average seeds before resampling; two seeds do not establish training-seed generality. No new optimizer updates and no inference speed requirement.')
    pp=R/'provenance/common-dense-quality-protocol-v0.json';assert not pp.exists();save(pp,p)
    files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp
    a=R/'exports/common-dense-quality-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
