"""Freeze known QK-Restore as a diagnostic control, not a proposed contribution."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    assert load(R/'results/fresh-word-dense-audit-v0/result.json')['status']=='verified'
    parent=load(R/'provenance/word-suffix-protocol-v0.json');jobs=[];refs=[];files={}
    for j in parent['jobs']:
        j=dict(j);j.update(name=f"restore-qk-train{j['k']}-seed{j['seed']}",phase='known_baseline');jobs.append(j)
    for ref in parent['references']:
        if ref['evaluation_k']!=0:continue
        prefix='results/cloud-word-counterfactual-evidence-v1' if ref['training_k']==0 else 'results/cloud-word-operator-swap-evidence-v0'
        f=R/prefix/ref['path'];assert sha(f)==ref['sha256'];files[ref['path']]=f
        refs.append(dict(**ref,kind='word'))
    nq=load(R/'provenance/common-dense-quality-protocol-v0.json')
    for j in nq['jobs']:
        if j['step']!=128:continue
        n='results/common-dense-quality-stage-v0/'+j['name']+'/result.json';f=R/'results/cloud-common-dense-quality-evidence-v0'/n;files[n]=f
        refs.append(dict(path=n,sha256=sha(f),training_k=j['k'],evaluation_k=0,seed=j['seed'],kind='nll'))
    assert len(jobs)==4 and len(refs)==8
    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','eval_qk_restore_baseline_v0.py','run_qk_restore_baseline_stage_v0.py','prepare_qk_restore_baseline_v0.py','report_qk_restore_baseline_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in names}
    data={n:sha(R/n) for n in ['data/32k-expanded-training-v0/config.json','data/32k-expanded-training-v0/train-calibration.npz','data/32k-expanded-training-v0/report.npz','data/word-counterfactual-v0/tasks.npz','data/word-counterfactual-v0/tasks.json','data/word-counterfactual-v0/manifest.json','docs/qk-restore-overlap-2026-09-16.md']}
    for ref in refs:data[ref['path']]=ref['sha256']
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,references=refs,source_sha256=sources,data_sha256=data,evaluation_gpu=parent['evaluation_gpu'],answer_token_ids=parent['answer_token_ids'],calibration_max_abs_error=1e-6,short_gate=dict(min_correct=56,min_pairs_both_correct=24,min_gain_over_no_context=24),long_gate=dict(min_correct=48,min_pairs_both_correct=20,min_gain_over_no_context=24),expected_task_predictions=768,expected_gradient_passes=0,optimizer_updates=0,maximum_seconds=1200,maximum_job_seconds=280,maximum_gpu_cost_usd_excluding_setup_storage=1200*.74/3600,scope='Known QK-Restore baseline only; overlap explicitly documented. Dense/K32-trained128-step LoRA checkpoints, bothseeds; after native NLL replay1e-6 zero only48 q_proj/k_proj LoRA B tensors, restoring effective pretrained Q/K exactly. Verify all144 other adapter tensors bitwise unchanged. Common dense evaluation. Reuse full-model common-dense word and9-window NLL references; new ablations on old32-background word development set only, no new independent confirmation or method novelty. Report both dense and sparse effects and their interaction, longword accuracy and PPL; do not infer training-time freezing outcome from posthoc reset. Same training weights saved on disk remain unchanged, ablated tensors exist only in memory. No fitting and no optimizer updates.')
    pp=R/'provenance/qk-restore-baseline-protocol-v0.json';assert not pp.exists();save(pp,p)
    for n in list(sources)+list(data):files.setdefault(n,R/n)
    files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/qk-restore-baseline-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
