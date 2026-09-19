"""Select a speed-qualified gentler route, then freeze a two-seed quality pilot."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,numpy as np,py_compile
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 candidates=[];audits={}
 for layout in ['m64','m128n32']:
  f=R/f'results/{layout}-density-cost-audit-v0/result.json';a=load(f);assert a['status']=='verified';audits[layout]=dict(path=f.relative_to(R).as_posix(),sha256=sha(f))
  for c in a['comparisons']:
   if c['k']>32 and c['eligible_for_calibration_training']:candidates.append(dict(layout=layout,k=c['k'],median_time_ratio=float(np.median(c['ratios']))))
 assert candidates,'No faster gentler configuration: do not launch quality training'
 # Prefer more retained blocks if speed-qualified, then lower measured ratio.
 chosen=sorted(candidates,key=lambda c:(-c['k'],c['median_time_ratio']))[0];layout=chosen['layout'];k=chosen['k']
 d=R/'data/gentle32k-v0';d.mkdir(exist_ok=False);cfg=load(R/'data/32k-expanded-training-v0/config.json');oldcfg=dict(cfg)
 source=(R/'scripts/run_expanded76.py').read_text(encoding='utf-8').replace("'run_expanded76.py'","'run_gentle32k.py'")
 source=source.replace("data/'config.json'","R/'data/gentle32k-v0/config.json'")
 source=source.replace('import run_flashmoba_realtext_precision as base',f'import run_flashmoba_realtext_precision as base\n from topk_{layout}_adapter import enable\n implementation=enable(base)')
 source=source.replace("out=a.output;out.mkdir(parents=True,exist_ok=False)","protocol=json.loads((R/'provenance/gentle32k-protocol.json').read_text())\n for n,h in protocol['source_sha256'].items():assert sha(R/n)==h,n\n assert implementation['sha256']==protocol['module_sha256']\n out=a.output;out.mkdir(parents=True,exist_ok=False)")
 source=source.replace("scope=cfg['scope'])","scope=cfg['scope'],implementation=implementation)")
 (R/'scripts/run_gentle32k.py').write_text(source,encoding='utf-8')
 cfg.update(methods=[k],prepared_utc=datetime.now(timezone.utc).isoformat(),selection_rule='Density/layout chosen using only previous full-update speed gates: maximum eligible K>32, then lower median paired time ratio. Fixed128 endpoint, fixed original .001-to-.0001 LR; no selection using this pilot quality.',scope=f'Exploratory Qwen2.5-0.5B LoRA adaptation on76 WikiText32K windows, K{k}/{layout},2 reused seeds,128 updates. Reuse verified dense128 parent controls; same initial params, data order and LR. Reuses old64 article tests; no independent confirmation claim.')
 cfg['sources_sha256']={n:sha(R/'scripts'/n) for n in ['run_gentle32k.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py']};save(d/'config.json',cfg)
 module=load(R/f'experiments/topk-{layout}-v0/build-result.json')
 p=dict(created_utc=cfg['prepared_utc'],chosen=chosen,considered=candidates,speed_audits=audits,steps=128,k=k,seeds=cfg['seeds'],maximum_seconds=2100,maximum_job_seconds=650,expected_counts=dict(scientific_updates=256,diagnostic_updates=6,task_predictions=576),module_sha256=module['module_sha256'],parent_audit_sha256=sha(R/'results/expanded76-audit-v0/result.json'),parent_config_sha256=sha(R/'data/32k-expanded-training-v0/config.json'),config_sha256=sha(d/'config.json'),source_sha256={n:sha(R/n) for n in ['scripts/run_gentle32k.py','scripts/run_gentle32k_stage.py','scripts/replay_gentle_dense_reference.py','scripts/run_expanded76.py',f'scripts/topk_{layout}_adapter.py',f'experiments/topk-{layout}-v0/build-result.json',f'experiments/topk-{layout}-v0/build/nsp_topk_{layout}_v0.so','scripts/density_tradeoff_math_gate.py','scripts/amp_recovery_state.py','scripts/chunked_lm_loss.py','scripts/run_flashmoba_realtext_precision.py']},scope=cfg['scope'],dense_reference='Reuse audited expanded76 dense128 predictions and training clocks, replay both dense parent calibrations before new training; verify data/config/seed/environment agreement. No repeat dense optimizer updates.',report_rule='Two fixed128 endpoints and one seed0 zero-update candidate. Original64 development articles only; no additional fresh256 evaluation in this stage. Stop after final reports, no automatic extension.')
 p['data_sha256']={n:sha(R/n) for n in ['data/32k-expanded-training-v0/train-calibration.npz','data/32k-expanded-training-v0/report.npz','data/32k-expanded-training-v0/tasks.json','data/32k-expanded-training-v0/tasks.npz','data/32k-expanded-training-v0/config.json']}
 save(R/'provenance/gentle32k-protocol.json',p)
 py_compile.compile(str(R/'scripts/run_gentle32k.py'),doraise=True);print(json.dumps(dict(chosen=chosen,protocol_sha256=sha(R/'provenance/gentle32k-protocol.json'))))
if __name__=='__main__':main()
