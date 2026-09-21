"""Freeze matched-checkpoint natural cloze validation and an operational base gate."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io,ast
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    pp=R/'provenance/lambada-natural-protocol-v0.json';assert not pp.exists()
    code=(R/'scripts/eval_pg19_quality_retry_v0.py').read_text(encoding='utf-8').replace('provenance/pg19-quality-retry-protocol-v0.json','provenance/lambada-natural-protocol-v0.json')
    start=code.index("  state['k']=0\n  event('common_dense_evaluation'");end=code.index('\n except Exception:',start)
    code=code[:start]+(R/'scripts/lambada_natural_eval_tail_v0.txt').read_text(encoding='utf-8')+code[end:]
    code=code.replace("  checkpoint=R/job['path']", "  identity=job['identity']\n  checkpoint=R/job['path']")
    code=code.replace("old=next(e for e in trained['evaluations'] if e['step']==step and e['split']=='calibration')['values']", "old=job['calibration_values']")
    ast.parse(code);path=R/'scripts/eval_lambada_natural_v0.py';assert not path.exists();path.write_text(code,encoding='utf-8')
    ctrl=(R/'scripts/run_fresh_word_dense_stage_v0.py').read_text(encoding='utf-8').replace('fresh-word-dense','lambada-natural').replace('fresh_word_dense','lambada_natural')
    ctrl=ctrl.replace("not (v['short_gate']['passed'] and v['long_gate'] and v['long_gate']['passed'])", "not v['base_gate']['passed']")
    ctrl=ctrl.replace("count in [5,133,p['expected_task_predictions']]", "count in [1024,p['expected_task_predictions']]")
    ast.parse(ctrl);path=R/'scripts/run_lambada_natural_stage_v0.py';assert not path.exists();path.write_text(ctrl,encoding='utf-8')
    parent=load(R/'provenance/pg19-quality-retry-protocol-v0.json');base=dict(parent['jobs'][0]);original=load(R/'results/cloud-expanded76-evidence-v0'/base['training_result'])
    base.update(name='base-ability',phase='base_gate',identity=original['identity'],calibration_values=next(e['values'] for e in original['evaluations'] if e['step']==0 and e['split']=='calibration'));jobs=[base]
    mp=load(R/'provenance/matched-restore-training-protocol-v0.json');mirror=R/'results/cloud-matched-restore-training-evidence-v0'
    for j in mp['jobs']:
        root='results/matched-restore-training-stage-v0/'+j['name'];v=load(mirror/root/'result.json')
        for restore in [False,True]:
            jobs.append(dict(name=j['name']+'-restore'+str(int(restore)),phase='natural_validation',k=0,training_k=j['training_k'],seed=j['seed'],step=128,path=root+'/checkpoint-128.pt',sha256=sha(mirror/root/'checkpoint-128.pt'),training_result=root+'/result.json',training_result_sha256=sha(mirror/root/'result.json'),restore_qk=restore,identity=v['identity'],calibration_values=next(e['common_dense_calibration_values'] for e in v['evaluations'] if e['step']==128)))
    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','prepare_lambada_natural_data_v0.py','prepare_lambada_natural_v0.py','lambada_natural_eval_tail_v0.txt','eval_lambada_natural_v0.py','run_lambada_natural_stage_v0.py','report_lambada_natural_v0.py','collect_lambada_natural_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in names};data={n:sha(R/n) for n in ['data/32k-expanded-training-v0/config.json','data/32k-expanded-training-v0/train-calibration.npz']}
    for f in (R/'data/lambada-natural-v0').iterdir():
        if f.is_file():data[f.relative_to(R).as_posix()]=sha(f)
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_sha256=sources,data_sha256=data,evaluation_gpu=parent['evaluation_gpu'],calibration_max_abs_error=1e-6,expected_task_predictions=5120,expected_gradient_passes=0,expected_calibration_nll_forwards=36,optimizer_updates=0,maximum_seconds=1800,maximum_job_seconds=240,maximum_gpu_cost_usd_excluding_setup_storage=1800*.74/3600,base_gate=dict(min_full_accuracy=.20,min_full_minus_short_accuracy=.05),primary=dict(endpoint=128,contrast='restored sparse vs original dense',accuracy_margin_pp=5,bootstrap_seed=2026091706,draws=20000,method='Average paired seed differences within each item and resample512 items. Pass only lower two-sided95% accuracy difference >-5pp; base gate must pass. Secondary original/symmetric-restored contrasts and summed target-word NLL. Conditional on seeds; unknown source-book grouping prevents independent book CI.'),scope=load(R/'data/lambada-natural-v0/manifest.json')['scope']+' Uses four freshly completed matched4090 checkpoint128 endpoints plus originalbase. Dense common operator. Before any new task scores, replay4 corresponding common-dense calibration windows <=1e-6. After replay zero48Q/K LoRA B only for restore jobs and verify other144tensors unchanged. First base full and last32context variants512each; if accuracy<20% or full-minus-short<5pp, archive and stop without editing task or running trained models. Otherwise8trained conditions full512each. Correctness requires all target subtoken argmax predictions match, using full vocabulary teacher-forced likelihood: any incorrect subtoken fails whole word. Ground-truth prefix is only scoring, not candidate-choice restriction. No training or32Kclaim, no selecting checkpoints based on these results; known restoration baseline, novelty not established.')
    save(pp,p);files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp
    a=R/'exports/lambada-natural-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
