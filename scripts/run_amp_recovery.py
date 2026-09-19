"""Bounded AMP LoRA pilot. Training never loads report tokens."""
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
 return {n:sha(R/'scripts'/n) for n in ['run_amp_recovery.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py']}
def main(a):
 import run_flashmoba_realtext_precision as base
 data=R/'data/flashmoba-amp-recovery-v0';cfg=json.loads((data/'config.json').read_text())
 out=a.output;out.mkdir(parents=True,exist_ok=False)
 started=utc();tick=time.perf_counter();rows=[];evaluations=[];gates=[];step=0;local_updates=0
 save(out/'config.json',cfg);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 def event(kind,**kw):
  x=dict(utc=utc(),event=kind);x.update(kw)
  with (out/'events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
  print(json.dumps(x),flush=True)
 try:
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
  assert train.shape==(64,16385) and cal.shape==(4,16385)
  order=np.random.default_rng(2026091562).permutation(64).tolist()
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
   loss=backward(train[order[i%64]]);norm=torch.nn.utils.clip_grad_norm_(list(params.values()),cfg['gradient_clip'],error_if_nonfinite=True)
   optimizer.step();scheduler.step();local_updates+=1;torch.cuda.synchronize()
   row=dict(step=i+1,window_index=order[i%64],train_nll=float(loss),gradient_norm=float(norm),lr=lr,seconds=time.perf_counter()-st,utc=utc())
   rows.append(row);event('step',**row)
  def checkpoint():
   save_checkpoint(out/f'checkpoint-{step}.pt',params,optimizer,scheduler,step,identity,dict(rows=rows,evaluations=evaluations,initial_sha256=initial_sha))
   save(out/'checkpoint-index.json',dict(step=step,path=f'checkpoint-{step}.pt',sha256=sha(out/f'checkpoint-{step}.pt'),utc=utc()))
  gates=[]
  if a.phase=='preflight':
   # Re-run arithmetic gates on this exact runner and environment before updates.
   state['reference']=True;rl=float(backward(train[0,:257],False));rg=gradients();state['reference']=False
   for k in [0,2]:
    state['k']=k;l=float(backward(train[0,:257],False));g=gradients();rel=float((g-rg).norm()/rg.norm())
    gate=dict(name='math_reference_gradient',k=k,nll_gap=abs(l-rl),gradient_relative_l2=rel,passed=abs(l-rl)<.05 and rel<.10)
    gates.append(gate);assert gate['passed'],gate
   del rg,g;state['k']=a.k
   l=float(backward(train[0,:2049],False));g=gradients();cl=float(backward(train[0,:2049],True));cg=gradients()
   rel=float((cg-g).norm()/g.norm());gate=dict(name='chunked_gradient',nll_gap=abs(l-cl),gradient_relative_l2=rel,passed=abs(l-cl)<.001 and rel<.02)
   gates.append(gate);assert gate['passed'],gate;del g,cg
   ids,_=inputs(train[0]);altered=ids.clone();altered[:,8192:]=torch.roll(altered[:,8192:],17,1)
   with torch.no_grad(),amp():
    prefix=model.model(ids,use_cache=False).last_hidden_state[:,:128].clone();other=model.model(altered,use_cache=False).last_hidden_state[:,:128]
   delta=float((prefix-other).abs().max());gates.append(dict(name='future_prefix',max_abs=delta,passed=delta==0));assert delta==0
   del ids,altered,prefix,other
   for i in range(4):
    update(i);step=i+1
    if step==2:checkpoint()
   expected={n:p.detach().cpu().clone() for n,p in params.items()};expected_losses=[x['train_nll'] for x in rows[2:]]
   save(out/'uninterrupted-trace.json',rows)
   step,extra=restore_checkpoint(out/'checkpoint-2.pt',params,optimizer,scheduler,identity);rows=extra['rows']
   for i in range(step,4):update(i);step=i+1
   errors={n:float((p.detach().cpu()-expected[n]).abs().max()) for n,p in params.items()}
   passed=all(torch.allclose(p.detach().cpu(),expected[n],atol=1e-6,rtol=1e-4) for n,p in params.items()) and all(abs(x['train_nll']-v)<.001 for x,v in zip(rows[2:],expected_losses))
   gates.append(dict(name='gpu_resume',max_parameter_abs_error=max(errors.values()),passed=passed));save(out/'gates.json',gates);assert passed,gates[-1]
   save(out/'gates.json',gates);checkpoint()
  elif a.phase=='report':
   selection=json.loads(a.selection.read_text());assert selection['config_sha256']==identity['config_sha256'] and selection['sources']==identity['sources']
   assert selection['selected_lrs'][str(a.k)]==a.lr
   assert a.resume is not None
   step,extra=restore_checkpoint(a.resume,params,optimizer,scheduler,identity);assert step==cfg['steps']
   assert sha(data/'report.npz')==cfg['report_sha256']
   report=np.load(data/'report.npz')['report'];evaluate('report',report)
   full=[nll(w,12288) for w in report];short=[nll(w[-4097:]) for w in report]
   assert all(math.isfinite(x) for x in full+short)
   save(out/'context-diagnostic.json',dict(utc=utc(),full_context_tail_nll=full,short_context_tail_nll=short,benefit_nats=[s-f for s,f in zip(short,full)],scope='Same final4096 targets, full16K versus last4K context. Positions reset for short input. Descriptive context-sensitivity diagnostic, not a long-range reasoning benchmark.'))
  else:
   if a.resume:
    step,extra=restore_checkpoint(a.resume,params,optimizer,scheduler,identity);rows=extra['rows'];evaluations=extra['evaluations']
   else:evaluate('calibration',cal)
   assert step<=cfg['steps']
   for i in range(step,cfg['steps']):
    update(i);step=i+1
    if step in cfg['calibration_steps']:evaluate('calibration',cal)
    if step in cfg['checkpoint_steps']:checkpoint()
   checkpoint()
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
 p.add_argument('--preflight',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--selection',type=Path);main(p.parse_args())
