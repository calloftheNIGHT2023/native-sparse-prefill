"""Calibration-only continuation gate, committed before stage-one task evaluation."""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 intent_path=R/'provenance/32k-extension-intent.json';intent=json.loads(intent_path.read_text());out=R/'logs/32k-extension-chain-v0.json';tick=time.monotonic()
 try:
  for n,h in intent['source_sha256'].items():assert sha(R/n)==h
  pp=R/'provenance/32k-continuation-protocol.json';assert sha(pp)==intent['parent_protocol_sha256'];old=json.loads(pp.read_text());parent=R/'results/32k-continuation-stage-v0'
  save(out,dict(status='waiting_for_parent_training',utc=utc(),intent_sha256=sha(intent_path)))
  while not all((parent/f'k{k}-seed{s}'/'result.json').exists() for k,s in old['run_order']):
   assert time.monotonic()-tick<3600
   if (parent/'result.json').exists():assert json.loads((parent/'result.json').read_text())['status']=='complete'
   time.sleep(10)
  results={f'k{k}-seed{s}':json.loads((parent/f'k{k}-seed{s}'/'result.json').read_text()) for k,s in old['run_order']}
  assert all(v['status']=='complete' and v['step']==128 for v in results.values())
  gains=[]
  for seed in old['seeds']:
   es=results[f'k32-seed{seed}']['evaluations'];nll=lambda step:next(e['mean_nll'] for e in es if e['split']=='calibration' and e['step']==step)
   gains.append(nll(96)-nll(128))
  passed=sum(gains)/len(gains)>=intent['minimum_mean_nll_gain'] and min(gains)>0
  decision=dict(status='waiting_for_parent_evidence' if passed else 'calibration_plateau_no_extension',utc=utc(),intent_sha256=sha(intent_path),gains_96_to_128=gains,mean_gain=sum(gains)/len(gains),gate_passed=passed,rule=intent['gate_rule'])
  save(out,decision)
  if not passed:return
  parents={}
  for key,v in results.items():
   k,s=map(int,[key.split('-')[0][1:],key.split('seed')[1]]);cp=parent/key/'checkpoint-128.pt'
   parents[key]=dict(k=k,seed=s,checkpoint=cp.relative_to(R).as_posix(),checkpoint_sha256=sha(cp),parent_training_seconds=v['training_seconds'],resume_identity=v['identity'],dense128_training_seconds=results[f'k0-seed{s}']['training_seconds'])
  p={n:old[n] for n in ['parent_config_sha256','task_data_path','task_metadata_sha256','task_tokens_sha256','report_sha256','seeds']}
  p.update(version=0,frozen_utc=utc(),intent_sha256=sha(intent_path),gate=decision,source_sha256=intent['source_sha256'],parents=parents,end_step=256,calibration_steps=[160,192,224,256],checkpoint_steps=[160,192,224,256],run_order=[[0,old['seeds'][0]],[32,old['seeds'][1]],[32,old['seeds'][0]],[0,old['seeds'][1]]],max_controller_seconds=3600,planned_new_updates=512,planned_task_forwards=1536,report_rule='Fixed four step256 endpoints, two sparse cumulative-time cuts paired with dense128 and two dense128 controls. No selection by reused report/task outcomes. Prior zero-update baselines remain in linked stage-one report.',schedule='Restore full step128 state, including nested parent identity, AdamW, scheduler, RNG, cursor. Continue same LR0.0001 floor to256 on same 32 windows. Eight total passes per lineage at256.',cost_scope=old['cost_scope'].replace('dense64','dense128'),claim_scope='Exploratory extension of four existing Qwen2.5-0.5B LoRA lineages, no new independent seeds. RACE64 and WikiText test are reused; no fresh confirmation or formal equivalence/novelty claim. Lower training cost with approximately comparable quality is the objective.')
  target=R/'provenance/32k-extension-protocol.json';assert not target.exists();save(target,p)
  decision['extension_protocol_sha256']=sha(target);save(out,decision)
  while not (R/'exports/32k-continuation-evidence-v0.tar.json').exists():
   assert time.monotonic()-tick<3600
   if (parent/'result.json').exists():assert json.loads((parent/'result.json').read_text())['status']=='complete'
   time.sleep(10)
  assert json.loads((parent/'result.json').read_text())['status']=='complete'
  running=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not running,'GPU busy; extension not launched concurrently'
  decision.update(status='running_extension',started_utc=utc());save(out,decision)
  with (R/'logs/32k-extension-controller-v0.log').open('w') as f:
   proc=subprocess.Popen([sys.executable,'-u',str(R/'scripts/run_32k_extension_stage.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT)
   decision['controller_pid']=proc.pid;save(out,decision)
   code=proc.wait(timeout=3700)
  decision.update(status='complete' if code==0 else 'extension_failed',finished_utc=utc(),returncode=code);save(out,decision)
 except Exception:
  save(out,dict(status='failed',utc=utc(),error=traceback.format_exc()));raise
if __name__=='__main__':main()
