"""Discriminating CPU checks: interrupted stochastic training vs continuous training."""
import copy,json,random,tempfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from amp_recovery_state import save_checkpoint,restore_checkpoint,lr_factor
from amp_recovery_selection import select_lrs
from chunked_lm_loss import chunked_head_backward
R=Path(__file__).resolve().parents[1]
def main():
 out=R/'results/flashmoba-amp-recovery-cpu-v1';out.mkdir(exist_ok=False)
 def setup():
  random.seed(12);np.random.seed(13);torch.manual_seed(14)
  m=torch.nn.Sequential(torch.nn.Linear(5,8),torch.nn.Tanh(),torch.nn.Dropout(.2)).double()
  head=torch.nn.Linear(8,7).double().requires_grad_(False)
  opt=torch.optim.AdamW(m.parameters(),lr=.01,foreach=False);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:lr_factor(s,2,8))
  return m,head,opt,sched
 def steps(m,head,opt,sched,start,end):
  losslist=[]
  for i in range(start,end):
   # Consume all RNG families, catching partial-RNG checkpoint implementations.
   x=torch.randn(1,11,5,dtype=torch.float64)*(random.random()+float(np.random.rand()));target=torch.randint(7,(11,))
   opt.zero_grad(set_to_none=True);h=m(x);loss=chunked_head_backward(h,head,target,3)
   torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();sched.step();losslist.append(float(loss))
  return losslist
 m,h,o,s=setup();first=steps(m,h,o,s,0,4);identity=dict(test='cpu',data='fixed-stochastic-stream')
 p=out/'checkpoint-4.pt';save_checkpoint(p,dict(m.named_parameters()),o,s,4,identity)
 full=steps(m,h,o,s,4,8);expected=copy.deepcopy(m.state_dict());expected_o=copy.deepcopy(o.state_dict());expected_s=copy.deepcopy(s.state_dict())
 m2,h2,o2,s2=setup();steps(m2,h2,o2,s2,0,1)
 step,extra=restore_checkpoint(p,dict(m2.named_parameters()),o2,s2,identity);assert step==4
 resumed=steps(m2,h2,o2,s2,step,8)
 assert full==resumed and all(torch.equal(v,m2.state_dict()[n]) for n,v in expected.items())
 def equal(a,b):
  if torch.is_tensor(a):return torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  return a==b
 assert equal(expected_o,o2.state_dict()) and equal(expected_s,s2.state_dict())
 rejected=False
 try:restore_checkpoint(p,dict(m2.named_parameters()),o2,s2,dict(test='wrong'))
 except AssertionError:rejected=True
 assert rejected
 cfg=json.loads((R/'data/flashmoba-amp-recovery-v0/config.json').read_text());mock=[]
 for k in cfg['methods']:
  for lr in cfg['learning_rates']:
   mock.append(dict(status='complete',phase='train',step=256,identity=dict(k=k,lr=lr,seed=cfg['seeds'][0]),initial_sha256='same',evaluations=[dict(split='calibration',step=256,mean_nll=1.0)]))
 assert select_lrs(list(reversed(mock)),cfg)=={str(k):min(cfg['learning_rates']) for k in cfg['methods']}
 bad=copy.deepcopy(mock);bad[0]['evaluations'].append(dict(split='report',step=256,mean_nll=.1))
 rejected=False
 try:select_lrs(bad,cfg)
 except AssertionError:rejected=True
 assert rejected
 bad=copy.deepcopy(mock);bad[0]['evaluations'][0]['mean_nll']=float('nan')
 rejected=False
 try:select_lrs(bad,cfg)
 except AssertionError:rejected=True
 assert rejected
 sel=cfg['selection'];assert set(sel['train']).isdisjoint(sel['report'])
 for i in sel['report']:assert not any(i*16385<b and (i+1)*16385>a for a,b in sel['prior_train_intervals'])
 result=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),tests=['continuous_vs_resumed_stochastic_loss_exact','model_parameters_exact','adam_state_exact','scheduler_state_exact','wrong_identity_rejected','selection_order_and_ties','report_contaminated_selection_rejected','nonfinite_selection_rejected','report_interval_disjointness'],cpu_updates=13,gpu_updates=0,gpu_validation='pending')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result))
if __name__=='__main__':main()
