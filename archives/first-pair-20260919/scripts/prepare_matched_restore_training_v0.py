"""Freeze complete matched training and posthoc restoration cost/quality replication."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    pp=R/'provenance/matched-restore-training-protocol-v0.json';assert not pp.exists()
    code=(R/'scripts/train_vo_only_v0.py').read_text(encoding='utf-8').replace('vo-training-protocol','matched-restore-training-protocol')
    start=code.index('  frozen={n:p for n,p in params.items()');end=code.index('  def digest(ps):',start)
    code=code[:start]+"  frozen={}\n  trainable=dict(params);assert len(trainable)==192 and sum(p.numel() for p in trainable.values())==1081344\n"+code[end:]
    code=code.replace("trainable_projection_names=['v_proj','o_proj'],trainable_parameters=540672", "trainable_projection_names=['q_proj','k_proj','v_proj','o_proj'],trainable_parameters=1081344")
    code=code.replace("def inputs(w):return torch.tensor(w[:-1][None],device='cuda'),torch.tensor(w[1:],device='cuda')", "def inputs(w):return torch.tensor(w[:-1][None],device='cuda',dtype=torch.long),torch.tensor(w[1:],device='cuda',dtype=torch.long)")
    code=code.replace('def nll(w):','def nll(w,target_start=0):').replace("range(0,len(target),cfg['chunk_size'])", "range(target_start,len(target),cfg['chunk_size'])").replace('return total/len(target)','return total/(len(target)-target_start)')
    start=code.index("  report=np.load(R/'data/32k-expanded-training-v0/report.npz')")
    end=code.index('\n except Exception:',start)
    tail=(R/'scripts/matched_restore_training_eval_tail_v0.txt').read_text(encoding='utf-8')
    code=code[:start]+tail+code[end:];ast.parse(code)
    path=R/'scripts/train_matched_restore_v0.py';assert not path.exists();path.write_text(code,encoding='utf-8')
    ctrl=(R/'scripts/run_vo_training_stage_v0.py').read_text(encoding='utf-8').replace('vo-training','matched-restore-training').replace('vo_training','matched_restore_training').replace('train_vo_only_v0.py','train_matched_restore_v0.py')
    ast.parse(ctrl);path=R/'scripts/run_matched_restore_training_stage_v0.py';assert not path.exists();path.write_text(ctrl,encoding='utf-8')
    parent=load(R/'provenance/vo-training-protocol-v0.json');jobs=[]
    for j in parent['jobs']:
        j=dict(j);j.update(name=j['name'].replace('vo-train','qkvo-train'),phase='matched_qkvo_training');jobs.append(j)
    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','prepare_matched_restore_training_v0.py','matched_restore_training_eval_tail_v0.txt','train_matched_restore_v0.py','run_matched_restore_training_stage_v0.py','report_matched_restore_training_v0.py','collect_matched_restore_training_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in names}
    data={n:sha(R/n) for n in ['data/32k-expanded-training-v0/config.json','data/32k-expanded-training-v0/train-calibration.npz','data/32k-expanded-training-v0/report.npz','data/fresh-word-dense-v0/tasks.npz','data/fresh-word-dense-v0/tasks.json','data/fresh-word-dense-v0/manifest.json','data/pg19-external-v0/windows.npz','data/pg19-external-v0/manifest.json','docs/qk-restore-overlap-2026-09-16.md']}
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_sha256=sources,data_sha256=data,evaluation_gpu=parent['evaluation_gpu'],answer_token_ids=load(R/'data/fresh-word-dense-v0/manifest.json')['label_token_ids'],calibration_max_abs_error=1e-6,steps_per_job=128,expected_optimizer_updates=512,expected_task_predictions=1064,expected_gradient_passes=8,expected_nll_forwards=504,maximum_seconds=4700,maximum_job_seconds=1150,maximum_gpu_cost_usd_excluding_setup_storage=4700*.74/3600,
      primary=dict(endpoint=128,comparison='sparse restored versus dense original',accuracy_margin_pp=5,ppl_ratio_margin=1.05,cost_saving_min_percent=5,bootstrap_seed=2026091704,draws=20000,cost='Each seed: complete process-to-training-finish plus timed restore+restored-adapter save for sparse, versus dense process-to-training-finish. Excludes scientific final quality evaluation for both. Also report full step, loop, original and restored process cost symmetrically. Require both positive savings and median>=5%.'),
      scope='Matched actual-training cost/quality replication, not novel method or independent new-data confirmation. All192QKVO LoRA tensors trainable; bothseeds start original dense step0 adapter checkpoint,128 dense orK32updates with identical76windows/order/AdamW/LR and nonreentrant checkpointing on4090. Two no-update32K gradient-repeat preflights perjob, native initialcal replay1e-6; checkpoint0/64/128 and64reload. At fixed128 save original checkpoint then time knownQKrestore and adapter serialization before any final qualityevaluation; evaluate original then restored weights in common dense attention on same exposed133word tasks and9WikiText/16PG19 books (full/tail/short). Both dense and sparse receive symmetric restoration diagnostics; primary restored sparse vs original dense. No eval affects training, no checkpoint selection. Counts512 scientificupdates,1064taskpredictions,504NLLforwards,8gradientchecks. Whole stage including final evaluation logged separately; actual invoice and broad generalization still unverified.')
    save(pp,p);files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp
    a=R/'exports/matched-restore-training-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
