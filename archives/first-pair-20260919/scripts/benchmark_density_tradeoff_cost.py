"""Diagnostic complete-update timing from a common dense64 checkpoint; no quality claim."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,json,math,random,time,traceback,functools
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from torch.nn.attention import sdpa_kernel,SDPBackend
from chunked_lm_loss import chunked_head_backward
from amp_recovery_state import save_checkpoint,restore_checkpoint,lr_factor
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
class LoRALinear(torch.nn.Module):
 def __init__(self,layer,rank,alpha):
  super().__init__();self.base=layer;self.scale=alpha/rank
  self.A=torch.nn.Parameter(torch.empty(rank,layer.in_features,device=layer.weight.device,dtype=layer.weight.dtype))
  self.B=torch.nn.Parameter(torch.zeros(layer.out_features,rank,device=layer.weight.device,dtype=layer.weight.dtype))
  torch.nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
 def forward(self,x):return self.base(x)+(x@self.A.T@self.B.T)*self.scale
def source_identity():
 return {n:sha(R/'scripts'/n) for n in ['run_32k_adaptation.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py']}
def main(a):
 import run_flashmoba_realtext_precision as base
 data=R/'data/32k-adaptation-v0';cfg=json.loads((data/'config.json').read_text())
 out=a.output;out.mkdir(parents=True,exist_ok=False)
 started=utc();tick=time.perf_counter();rows=[];evaluations=[];gates=[];step=0;local_updates=0
 save(out/'config.json',cfg);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 def event(kind,**kw):
  x=dict(utc=utc(),event=kind);x.update(kw)
  with (out/'events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
  print(json.dumps(x),flush=True)
 try:
  protocol_path=R/'provenance/density-tradeoff-cost-protocol.json'
  protocol=json.loads(protocol_path.read_text())
  for name,h in protocol['source_sha256'].items():assert sha(R/name)==h
  assert a.mode in protocol['methods'] and a.round in range(3)
  assert source_identity()==cfg['sources_sha256']
  assert a.k in cfg['methods'] and a.seed in cfg['seeds'] and a.lr in cfg['learning_rates']
  assert torch.cuda.is_available() and sha(Path(base.flash_moba_cuda.__file__))==cfg['extension_sha256']
  assert cfg['deterministic_backward'] is True
  torch.use_deterministic_algorithms(True)
  base.flash_moba_attn_varlen_func=functools.partial(base.flash_moba_attn_varlen_func,deterministic=True)
  assert sha(data/'train-calibration.npz')==cfg['data_sha256']
  for x in json.loads((R/'data/flashmoba-qwen-precision-v0/manifest.json').read_text())['files']:
   if x['path'].startswith('model/'):assert sha(R/'data/flashmoba-qwen-precision-v0'/x['path'])==x['sha256']
  identity=dict(k=a.k,seed=a.seed,lr=a.lr,config_sha256=sha(data/'config.json'),sources=source_identity(),extension_sha256=cfg['extension_sha256'])
  environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()))
  if a.phase!='preflight':
   gate=json.loads((a.preflight/'result.json').read_text());assert gate['status']=='complete' and gate['phase']=='preflight'
   assert gate['identity']['k']==a.k and gate['identity']['sources']==identity['sources'] and gate['identity']['config_sha256']==identity['config_sha256']
   assert gate['environment']==environment
  arrays=np.load(data/'train-calibration.npz');train=arrays['train'];cal=arrays['calibration']
  assert train.shape==(32,32769) and cal.shape==(4,32769)
  order=np.random.default_rng(2026091662).permutation(32).tolist()
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
  random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
  model,info=AutoModelForCausalLM.from_pretrained(R/cfg['model_path'],local_files_only=True,trust_remote_code=False,
   use_safetensors=True,attn_implementation='eager',dtype=torch.float32,output_loading_info=True)
  assert not info['missing_keys'] and not info['unexpected_keys'] and not info.get('mismatched_keys')
  model=model.cuda().eval();model.requires_grad_(False);torch.manual_seed(a.seed)
  for layer in model.model.layers:
   for n in ['q_proj','k_proj','v_proj','o_proj']:setattr(layer.self_attn,n,LoRALinear(getattr(layer.self_attn,n),8,16))
  params={n:p for n,p in model.named_parameters() if p.requires_grad};assert sum(p.numel() for p in params.values())==1081344
  initial_sha=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in params.values())).hexdigest()
  state=dict(k=a.k,reference=False);base.LOCKED_POOL_CONFIG=(32,4,3)
  def amp():return torch.autocast('cuda',dtype=torch.bfloat16)
  def attention(module,query,key,value,attention_mask,scaling=None,dropout=0.,**kw):
   assert query.shape[0]==1 and query.shape[-2]==key.shape[-2] and dropout==0
   q,k,v=[x[0].transpose(0,1).to(torch.bfloat16).contiguous() for x in (query,key,value)]
   if state['reference']:
    with torch.autocast('cuda',enabled=False):
     qf=q.transpose(0,1)[None].float();kf=k.transpose(0,1)[None].float().repeat_interleave(q.shape[1]//k.shape[1],dim=1);vf=v.transpose(0,1)[None].float().repeat_interleave(q.shape[1]//v.shape[1],dim=1)
     mask=torch.ones(q.shape[0],k.shape[0],device=q.device,dtype=torch.bool).tril()
     y=(torch.softmax((qf@kf.transpose(-1,-2)*scaling).masked_fill(~mask,float('-inf')),dim=-1)@vf).to(torch.bfloat16)
    return y.transpose(1,2).to(query.dtype),None
   if state['k']==0:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):y=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True,dropout_p=0.,scale=scaling,enable_gqa=True)
    return y.transpose(1,2).to(query.dtype),None
   return base.sparse(q,k,v,128,state['k'],'fp32',scaling)[None].to(query.dtype),None
  ALL_ATTENTION_FUNCTIONS.register('amp_recovery',attention);model.config._attn_implementation='amp_recovery'
  optimizer=torch.optim.AdamW(list(params.values()),lr=a.lr,betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'],foreach=False,fused=False)
  scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:lr_factor(s,cfg['warmup_steps'],cfg['steps']))
  def inputs(w):return torch.tensor(w[:-1][None],device='cuda'),torch.tensor(w[1:],device='cuda')
  def backward(w,chunk=True):
   optimizer.zero_grad(set_to_none=True);ids,target=inputs(w)
   if chunk:
    with amp():h=model.model(ids,use_cache=False).last_hidden_state
    loss=chunked_head_backward(h,model.lm_head,target,cfg['chunk_size'],amp)
   else:
    with amp():loss=F.cross_entropy(model(ids,use_cache=False).logits[0].float(),target)
    loss.backward();loss=loss.detach()
   assert bool(torch.isfinite(loss))
   return loss
  def gradients():return torch.cat([p.grad.detach().float().flatten() for p in params.values()])
  @torch.no_grad()
  def nll(w,target_start=0):
   ids,target=inputs(w)
   with amp():h=model.model(ids,use_cache=False).last_hidden_state
   total=0.
   for i in range(target_start,len(target),cfg['chunk_size']):
    with amp():logits=model.lm_head(h[:,i:i+cfg['chunk_size']])
    total+=float(F.cross_entropy(logits[0].float(),target[i:i+cfg['chunk_size']],reduction='sum'))
   return total/(len(target)-target_start)
  def evaluate(split,windows):
   torch.cuda.synchronize();st=time.perf_counter();v=[nll(w) for w in windows];torch.cuda.synchronize()
   assert all(math.isfinite(x) for x in v)
   row=dict(split=split,step=step,values=v,mean_nll=float(np.mean(v)),seconds=time.perf_counter()-st,utc=utc())
   evaluations.append(row);save(out/'evaluations.json',evaluations);event('evaluation',**row)
  def update(i):
   nonlocal local_updates
   torch.cuda.synchronize();st=time.perf_counter();lr=optimizer.param_groups[0]['lr']
   loss=backward(train[order[i%32]]);norm=torch.nn.utils.clip_grad_norm_(list(params.values()),cfg['gradient_clip'],error_if_nonfinite=True)
   optimizer.step();scheduler.step();local_updates+=1;torch.cuda.synchronize()
   row=dict(step=i+1,window_index=order[i%32],train_nll=float(loss),gradient_norm=float(norm),lr=lr,seconds=time.perf_counter()-st,utc=utc())
   rows.append(row);event('step',**row)
  def checkpoint():
   save_checkpoint(out/f'checkpoint-{step}.pt',params,optimizer,scheduler,step,identity,dict(rows=rows,evaluations=evaluations,initial_sha256=initial_sha))
   save(out/'checkpoint-index.json',dict(step=step,path=f'checkpoint-{step}.pt',sha256=sha(out/f'checkpoint-{step}.pt'),utc=utc()))
  assert a.resume and sha(a.resume)==protocol['parent_checkpoint_sha256']
  step,extra=restore_checkpoint(a.resume,params,optimizer,scheduler,identity);assert step==64
  original=next(e for e in extra['evaluations'] if e['split']=='calibration' and e['step']==64)
  replay=[nll(w) for w in cal];replay_error=max(abs(x-y) for x,y in zip(replay,original['values']));assert replay_error<=1e-6
  from density_tradeoff_math_gate import check
  gate=check(a.mode,base);gates=[gate]
  step,extra=restore_checkpoint(a.resume,params,optimizer,scheduler,identity)
  before=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in params.values())).hexdigest()
  state['k']=a.mode;rows=[];evaluations=[]
  optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  for i in range(64,64+protocol['warmups']+protocol['samples']):
   update(i);step=i+1
  after=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in params.values())).hexdigest()
  measured=rows[protocol['warmups']:];assert len(measured)==protocol['samples'] and before!=after
  measured_median=float(np.median([x['seconds'] for x in measured]))
  save_checkpoint(out/'diagnostic-final.pt',params,optimizer,scheduler,step,dict(parent_identity=identity,protocol_sha256=sha(protocol_path),mode=a.mode,round=a.round,diagnostic=True),dict(rows=rows,parent_checkpoint_sha256=sha(a.resume)))
  save(out/'measurement.json',dict(mode=a.mode,round=a.round,median_seconds=measured_median,measured_seconds=[x['seconds'] for x in measured],warmup_seconds=[x['seconds'] for x in rows[:protocol['warmups']]],parent_replay_error=replay_error,parent_params_sha256=before,final_params_sha256=after,diagnostic_checkpoint_sha256=sha(out/'diagnostic-final.pt'),protocol_sha256=sha(protocol_path),optimizer_updates=local_updates,peak_gpu_bytes=torch.cuda.max_memory_allocated(),gate=gate,scope=protocol['scope']))
  result=dict(status='complete',phase=a.phase,identity=identity,environment=environment,started_utc=started,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,step=step,
   optimizer_updates_this_process=local_updates,
   initial_sha256=initial_sha,training_seconds=sum(x['seconds'] for x in rows),trace=rows,evaluations=evaluations,gates=gates,peak_gpu_bytes=torch.cuda.max_memory_allocated(),scope=cfg['scope'])
 except Exception:
  result=dict(status='failed',phase=a.phase,step=step,optimizer_updates_this_process=local_updates,started_utc=started,finished_utc=utc(),trace=rows,evaluations=evaluations,gates=gates,error=traceback.format_exc());event('failure',error=result['error'])
 save(out/'result.json',result)
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','train','report'],required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--k',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--lr',type=float,required=True)
 p.add_argument('--mode',type=int,required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--preflight',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--selection',type=Path);main(p.parse_args())
