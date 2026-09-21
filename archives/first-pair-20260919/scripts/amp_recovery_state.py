"""Atomic local checkpoints; load only trusted files created by this project."""
import os,random
from pathlib import Path
import numpy as np
import torch
def save_checkpoint(path,params,optimizer,scheduler,step,identity,extra=None):
 path=Path(path);tmp=path.with_suffix('.tmp')
 state=dict(version=1,params={n:p.detach().cpu().clone() for n,p in params.items()},optimizer=optimizer.state_dict(),
  scheduler=scheduler.state_dict(),step=step,data_cursor=step,identity=identity,extra=extra or {},
  rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
 with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def restore_checkpoint(path,params,optimizer,scheduler,identity):
 state=torch.load(path,map_location='cpu',weights_only=False)
 assert state['identity']==identity and state['step']==state['data_cursor']
 assert set(state['params'])==set(params)
 with torch.no_grad():
  for n,p in params.items():p.copy_(state['params'][n])
 optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
 random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch'])
 if state['rng']['cuda'] is not None:torch.cuda.set_rng_state_all(state['rng']['cuda'])
 return state['step'],state['extra']
def lr_factor(step,warmup,total):
 if step<warmup:return (step+1)/warmup
 return max(.1,1-.9*(step-warmup)/max(1,total-warmup))
