"""Freeze a stage-held-out split without reading any model outcomes."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=R/'data/flashmoba-amp-recovery-v0';out.mkdir(exist_ok=False)
 old=json.loads((R/'data/flashmoba-pool-training-v0/config.json').read_text())
 tok=AutoTokenizer.from_pretrained(R/old['model_path'],local_files_only=True)
 streams={};sources=[]
 for split in ['train','validation']:
  p=R/f'data/flashmoba-pool-training-v0/{split}.txt'
  assert sha(p)==next(x['text_sha256'] for x in old['sources'] if x['split']==split)
  streams[split]=np.asarray(tok(p.read_text(encoding='utf-8'),add_special_tokens=False)['input_ids'],dtype=np.int64)
  sources.append(dict(split=split,path=p.relative_to(R).as_posix(),sha256=sha(p),tokens=len(streams[split])))
 width=16385;blocked=[]
 for name,length in [('flashmoba-pool-training-v0',2048),('flashmoba-backward-training-v0',8192)]:
  m=json.loads((R/f'results/{name}/input-manifest.json').read_text())
  blocked.extend((i*(length+1),(i+1)*(length+1)) for i in m['selected']['train']['indices'])
 prior=json.loads((R/'data/flashmoba-quality-cost-v0/config.json').read_text())
 blocked.extend((i*8193,(i+1)*8193) for i in prior['selection']['train']['indices'])
 n=len(streams['train'])//width
 allowed=[i for i in range(n) if not any(i*width<b and (i+1)*width>a for a,b in blocked)]
 rng=np.random.default_rng(2026091559);assert len(allowed)>=12
 report=sorted(rng.choice(allowed,12,replace=False).tolist())
 candidates=[i for i in range(n) if i not in report];assert len(candidates)>=64
 train=rng.choice(candidates,64,replace=False).tolist()
 calibration=[0,1,2,3]
 def windows(split,indices):return np.stack([streams[split][i*width:(i+1)*width] for i in indices])
 np.savez(out/'train-calibration.npz',train=windows('train',train),calibration=windows('validation',calibration))
 np.savez(out/'report.npz',report=windows('train',report))
 cfg=dict(version=0,prepared_utc=datetime.now(timezone.utc).isoformat(),length=16384,chunk_size=256,
  model_path=old['model_path'],model_revision=old['model_revision'],dataset_revision=old['dataset_revision'],
  steps=256,learning_rates=[3e-5,1e-4,3e-4],seeds=[2026091560,2026091561],methods=[0,4,16],
  warmup_steps=16,calibration_steps=[0,64,128,256],checkpoint_steps=[2,64,128,256],
  lora_rank=8,lora_alpha=16,betas=[.9,.999],eps=1e-8,weight_decay=0.0,gradient_clip=1.0,
  extension_sha256='72c3fda9e7d4da701bfc76b0e32f4d80fdbeec70d462c83b87b97f51d887ed1c',
  data_sha256=sha(out/'train-calibration.npz'),report_sha256=sha(out/'report.npz'),
  selection=dict(train=train,report=report,calibration=calibration,report_available=len(allowed),prior_train_intervals=blocked),
  sources=sources,quality_margin_nats=.03,time_ratio_screen=.95,maximum_controller_seconds=7200,
  selection_rule='For each method choose lowest mean calibration NLL at step256 on seed0; tie chooses lower LR. Freeze choices before any report evaluation. Repeat only selected LR on seed1; evaluate all three methods on both seeds.',
  scope='Dense-pretrained 0.5B LoRA adaptation, not native full-parameter sparse pretraining. Stage-held-out train-corpus report intervals exclude known Qwen pilot training windows; historical other-model exposure and pretrained contamination are not excluded. Calibration is reused development data. No claims of globally untouched benchmark or novel ML contribution.')
 (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 (out/'source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps(dict(status='prepared',train_windows=64,report_windows=12,report_available=len(allowed),data_sha256=cfg['data_sha256'])))
if __name__=='__main__':main()
