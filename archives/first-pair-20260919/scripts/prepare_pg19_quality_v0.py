"""Freeze external natural-text evaluation before any model output on PG19."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io,ast
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    pp=R/'provenance/pg19-quality-protocol-v0.json';assert not pp.exists()
    code=(R/'scripts/eval_common_dense_quality_v0.py').read_text(encoding='utf-8').replace('provenance/common-dense-quality-protocol-v0.json','provenance/pg19-quality-protocol-v0.json')
    old=(R/'scripts/eval_qk_restore_baseline_v0.py').read_text(encoding='utf-8');block=old[old.index('  before={n:p.detach()'):old.index('  report=np.load')]
    ablation="  ablation=None\n  if job['restore_qk']:\n"+'\n'.join(' '+s for s in block.rstrip().splitlines())+'\n'
    anchor="  assert sha(data/'report.npz')==cfg['report_sha256']\n  report=np.load(data/'report.npz')['report'];assert report.shape==(9,32769)\n"
    assert code.count(anchor)==1
    code=code.replace(anchor,ablation+"  report=np.load(R/'data/pg19-external-v0/windows.npz')['windows'];assert report.shape==(16,32769)\n")
    code=code.replace("result=dict(status='complete',job=job", "result=dict(status='complete',ablation=ablation,job=job").replace('nll_forwards=31','nll_forwards=52')
    path=R/'scripts/eval_pg19_quality_v0.py';assert not path.exists();ast.parse(code);path.write_text(code,encoding='utf-8')
    code=(R/'scripts/run_common_dense_quality_stage_v0.py').read_text(encoding='utf-8').replace('common-dense-quality','pg19-quality').replace('common_dense_quality','pg19_quality')
    path=R/'scripts/run_pg19_quality_stage_v0.py';assert not path.exists();ast.parse(code);path.write_text(code,encoding='utf-8')
    old=load(R/'provenance/common-dense-quality-protocol-v0.json');jobs=[]
    j=dict(next(j for j in old['jobs'] if j['step']==0));j.update(name='base',phase='external_quality',restore_qk=False);jobs.append(j)
    for k,seed in [(0,2026091660),(32,2026091660),(32,2026091661),(0,2026091661)]:
        for restore in [False,True]:
            j=dict(next(j for j in old['jobs'] if j['step']==128 and j['k']==k and j['seed']==seed));j.update(name=f'train{k}-seed{seed}-restore{int(restore)}',phase='external_quality',restore_qk=restore);jobs.append(j)
    assert len(jobs)==9
    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','prepare_pg19_external_data_v0.py','prepare_pg19_quality_v0.py','eval_pg19_quality_v0.py','run_pg19_quality_stage_v0.py','report_pg19_quality_v0.py','collect_pg19_quality_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in names}
    data={n:sha(R/n) for n in ['data/32k-expanded-training-v0/config.json','data/32k-expanded-training-v0/train-calibration.npz','data/pg19-external-v0/official-test-listing.json','data/pg19-external-v0/windows.npz','data/pg19-external-v0/manifest.json','docs/qk-restore-overlap-2026-09-16.md']}
    for f in (R/'data/pg19-external-v0/raw').glob('*.txt'):data[f.relative_to(R).as_posix()]=sha(f)
    manifest=load(R/'data/pg19-external-v0/manifest.json')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_sha256=sources,data_sha256=data,evaluation_gpu=old['evaluation_gpu'],calibration_max_abs_error=1e-6,expected_task_predictions=0,expected_gradient_passes=0,expected_nll_forwards=468,optimizer_updates=0,maximum_seconds=2100,maximum_job_seconds=225,maximum_gpu_cost_usd_excluding_setup_storage=2100*.74/3600,
      primary=dict(contrast='restored sparse vs original dense at128',metrics=['full_nll','full_context_tail_nll'],ppl_ratio_noninferiority_upper=1.05,bootstrap_draws=20000,bootstrap_seed=2026091703,method='Average seed paired differences within eachbook; bootstrap16 books jointly for both seeds. For each co-primary metric require upper two-sided95% bound of exp(meanNLLdifference)<=1.05; report per-seed and all9 models. No outcome-based checkpoint/book/margin changes.'),
      scope=manifest['scope']+' Common dense evaluation, base plus dense/K32 original and knownQK-restored checkpoints from both existing seeds. Four native calibration NLL replay checks before intervention; zero48Q/K LoRAB tensors in memory only and verify144other tensors unchanged. Eachbook: full32K NLL, last4K targets with full32K and short4K context. Context benefit secondary, no input-specific tuning. 468 NLL forwards, no training, no task-accuracy predictions. Passing both5% PPL criteria is evidence only for this prespecified external-corpus sample, not paper sufficiency or evidence of originality/cost-to-convergence; unknown base-pretraining exposure and two-seed limits remain.')
    save(pp,p);files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp
    a=R/'exports/pg19-quality-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
