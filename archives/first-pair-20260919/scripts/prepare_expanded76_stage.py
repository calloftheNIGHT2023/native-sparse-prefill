"""Freeze a fresh four-trajectory, equal-token expanded-data pilot before launch."""
import json,hashlib,shutil,py_compile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 d=R/'data/32k-expanded-training-v0';assert not (d/'config.json').exists()
 old=json.loads((R/'data/32k-adaptation-v0/config.json').read_text());cfg=dict(old)
 a=np.load(d/'train-calibration.npz');b=np.load(R/'data/32k-adaptation-v0/train-calibration.npz')
 assert a['train'].shape==(76,32769) and np.array_equal(a['calibration'],b['calibration'])
 assert np.array_equal(a['train'][old['selection']['train']],b['train'])
 for name,src in [('report.npz','data/32k-separator-replay-v0/report.npz'),('tasks.npz','data/32k-continuation-tasks-v0/tasks.npz'),('tasks.json','data/32k-continuation-tasks-v0/tasks.json')]:shutil.copyfile(R/src,d/name)
 source=(R/'scripts/run_32k_adaptation.py').read_text()
 for before,after in [("'run_32k_adaptation.py'","'run_expanded76.py'"),("data/32k-adaptation-v0","data/32k-expanded-training-v0"),("train.shape==(32,32769)","train.shape==(76,32769)"),(".permutation(32)",".permutation(76)"),("i%32","i%76"),("cfg['warmup_steps'],cfg['steps']","cfg['warmup_steps'],cfg['lr_schedule_steps']")]:
  assert before in source,before;source=source.replace(before,after)
 # The old 257-token gate is dense-equivalent; add genuine K32 sparse math.
 source=source.replace("# Re-run arithmetic gates on this exact runner and environment before updates.","# Re-run original arithmetic gates plus genuinely sparse routing.\n   from density_tradeoff_math_gate import check\n   gates.append(check(a.k,base))")
 (R/'scripts/run_expanded76.py').write_text(source,encoding='utf-8')
 cfg.update(prepared_utc=datetime.now(timezone.utc).isoformat(),steps=128,lr_schedule_steps=64,learning_rates=[.001],calibration_steps=[0,32,64,96,128],checkpoint_steps=[0,2,32,64,96,128],data_sha256=sha(d/'train-calibration.npz'),report_sha256=sha(d/'report.npz'),task_metadata_sha256=sha(d/'tasks.json'),task_tokens_sha256=sha(d/'tasks.npz'),selection=dict(train=list(range(76)),calibration=list(range(4)),report=list(range(9))),selection_rule='Fixed LR .001 inherited from the previous symmetric search. Original 64-update schedule then .0001 floor; fixed128 endpoint, no checkpoint or LR selection on new outcomes.',scope='Exploratory expanded-data replication: Qwen2.5-0.5B dense-pretrained Q/K/V/O LoRA,76 packed WikiText32K windows,128 equal-token updates. Reuses prior two initialization seeds and prior WikiText/RACE64 evaluation sets; not new independent seeds, fresh test confirmation, native full-parameter pretraining, or a statistical equivalence claim.')
 cfg['sources_sha256']={n:sha(R/'scripts'/n) for n in ['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py']}
 save(d/'config.json',cfg)
 names=['scripts/run_expanded76.py','scripts/run_expanded76_stage.py','scripts/density_tradeoff_math_gate.py','scripts/amp_recovery_state.py','scripts/chunked_lm_loss.py','scripts/run_flashmoba_realtext_precision.py']
 p=dict(created_utc=cfg['prepared_utc'],question='Does the speed-quality tradeoff persist with76 training windows instead of repeatedly cycling32?',run_order=[[0,2026091660],[32,2026091661],[32,2026091660],[0,2026091661]],steps=128,methods=[0,32],seeds=cfg['seeds'],lr=.001,maximum_seconds=3600,maximum_job_seconds=650,source_sha256={n:sha(R/n) for n in names},data_sha256={str(f.relative_to(R)).replace('\\','/'):sha(f) for f in d.iterdir() if f.is_file()},expected_scientific_updates=512,expected_diagnostic_updates=12,expected_task_predictions=1152,scope=cfg['scope'],comparison='Dense and sparse start from pretrained weights with matching LoRA seeds; identical text order, optimizer and fixed schedule. Compare current dense/sparse endpoints primarily. Prior32-window curves are historical descriptive controls, with changed sample order as well as coverage.',report_rule='Four fixed128 endpoints and two seed0 zero-update baselines. No test-based selection. No automatic extension.',hardware_limit='Unmodified K48/K64 topk branch requests106496 shared-memory bytes vs101376 opt-in limit on current Ada; excluded from this stage, original failures retained.')
 save(R/'provenance/expanded76-protocol.json',p)
 for n in names:py_compile.compile(str(R/n),doraise=True)
 print(json.dumps(dict(protocol_sha256=sha(R/'provenance/expanded76-protocol.json'),config_sha256=sha(d/'config.json'))))
if __name__=='__main__':main()
