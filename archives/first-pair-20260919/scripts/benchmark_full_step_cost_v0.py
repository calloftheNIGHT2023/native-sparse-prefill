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
def source_identity(k):
 return {n:sha(R/'scripts'/n) for n in [('run_gentle32k.py' if k==48 else 'run_expanded76.py'),'amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py']}
def main(a):
 import run_flashmoba_realtext_precision as base
 protocol=json.loads((R/'provenance/full-step-cost-protocol-v0.json').read_text());job=next(x for x in protocol['jobs'] if x['name']==a.job)
 a.k=job['k'];a.seed=job['seed'];a.lr=.001;a.phase=job['phase']
 for n,h in {**protocol['source_sha256'],**protocol['data_sha256']}.items():assert sha(R/n)==h,n
 if a.k==48:
  from topk_m64_adapter import enable
  enable(base)
 data=R/'data/32k-expanded-training-v0';config_path=R/('data/gentle32k-v0/config.json' if a.k==48 else 'data/32k-expanded-training-v0/config.json');cfg=json.loads(config_path.read_text())
 out=a.output;out.mkdir(parents=True,exist_ok=False)
 started=utc();tick=time.perf_counter();rows=[];evaluations=[];gates=[];step=0;local_updates=0
 save(out/'config.json',cfg);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 def event(kind,**kw):
  x=dict(utc=utc(),event=kind);x.update(kw)
  with (out/'events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
  print(json.dumps(x),flush=True)
 try:
  assert source_identity(a.k)==cfg['sources_sha256']
  assert a.k in cfg['methods'] and a.seed in cfg['seeds'] and a.lr in cfg['learning_rates']
  assert torch.cuda.is_available() and sha(Path(base.flash_moba_cuda.__file__))==cfg['extension_sha256']
  assert cfg['deterministic_backward'] is True
  torch.use_deterministic_algorithms(True)
  base.flash_moba_attn_varlen_func=functools.partial(base.flash_moba_attn_varlen_func,deterministic=True)
  assert sha(data/'train-calibration.npz')==cfg['data_sha256']
  for x in json.loads((R/'data/flashmoba-qwen-precision-v0/manifest.json').read_text())['files']:
   if x['path'].startswith('model/'):assert sha(R/'data/flashmoba-qwen-precision-v0'/x['path'])==x['sha256']
  identity=dict(k=a.k,seed=a.seed,lr=a.lr,config_sha256=sha(config_path),sources=source_identity(a.k),extension_sha256=cfg['extension_sha256'])
  environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()))
  arrays=np.load(data/'train-calibration.npz');train=arrays['train'];cal=arrays['calibration']
  assert train.shape==(76,32769) and cal.shape==(4,32769)
  order=np.random.default_rng(2026091662).permutation(76).tolist()
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
  checkpoint=R/job['path'];assert sha(checkpoint)==job['sha256']
  assert sha(R/job['training_result'])==job['training_result_sha256']
  trained=json.loads((R/job['training_result']).read_text());assert {n:v for n,v in trained['environment'].items() if n!='gpu'}=={n:v for n,v in environment.items() if n!='gpu'}
  assert environment['gpu']==protocol['evaluation_gpu']
  c=torch.load(checkpoint,map_location='cpu',weights_only=False);step=c['step'];assert step==job['step'] and c['identity']==identity
  assert c['data_cursor']==step and set(c['params'])==set(params)
  with torch.no_grad():
   for n,p in params.items():p.copy_(c['params'][n])
  del c
  def inputs(w):return torch.tensor(w[:-1][None],device='cuda'),torch.tensor(w[1:],device='cuda')
  @torch.no_grad()
  def nll(w):
   ids,target=inputs(w)
   with amp():h=model.model(ids,use_cache=False).last_hidden_state
   total=0.
   for i in range(0,len(target),cfg['chunk_size']):
    with amp():logits=model.lm_head(h[:,i:i+cfg['chunk_size']])
    total+=float(F.cross_entropy(logits[0].float(),target[i:i+cfg['chunk_size']],reduction='sum'))
   return total/len(target)
  vals=[nll(w) for w in cal]
  old=next(e for e in trained['evaluations'] if e['step']==step and e['split']=='calibration')['values']
  error=max(abs(x-y) for x,y in zip(vals,old));assert error<=protocol['calibration_max_abs_error']
  event('calibration_replay',step=step,max_abs_error=error)
  assert a.k==0 and model.config.attention_dropout==0
  optimizer=torch.optim.AdamW(list(params.values()),lr=.001,betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'],foreach=False,fused=False)
  scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:lr_factor(s,cfg['warmup_steps'],cfg['lr_schedule_steps']))
  step,extra=restore_checkpoint(checkpoint,params,optimizer,scheduler,identity);assert step==128
  source_step=step;state['k']=job['training_k'];assert state['k'] in [0,32]
  model.train();model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  assert model.model.gradient_checkpointing
  new_identity=dict(parent=identity,training_k=state['k'],protocol_sha256=sha(R/'provenance/full-step-cost-protocol-v0.json'),gradient_checkpointing='nonreentrant')
  save_checkpoint(out/'checkpoint-128.pt',params,optimizer,scheduler,step,new_identity)
  rows=[];torch.cuda.reset_peak_memory_stats();loop_start=time.perf_counter()
  for i in range(source_step,source_step+protocol['steps_per_job']):
   torch.cuda.synchronize();st=time.perf_counter();wi=order[i%76];lr=optimizer.param_groups[0]['lr']
   optimizer.zero_grad(set_to_none=True);ids,target=inputs(train[wi])
   with amp():h=model.model(ids,use_cache=False).last_hidden_state
   loss=chunked_head_backward(h,model.lm_head,target,cfg['chunk_size'],amp)
   norm=torch.nn.utils.clip_grad_norm_(list(params.values()),cfg['gradient_clip'],error_if_nonfinite=True)
   assert bool(torch.isfinite(loss)) and bool(torch.isfinite(norm))
   optimizer.step();scheduler.step();torch.cuda.synchronize();elapsed=time.perf_counter()-st
   local_updates+=1;step=i+1
   row=dict(step=step,window_index=wi,input_sha256=hashlib.sha256(train[wi].tobytes()).hexdigest(),training_k=state['k'],loss=float(loss),gradient_norm=float(norm),lr=lr,seconds=elapsed,utc=utc());rows.append(row)
   with (out/'training-steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   event('optimizer_step',**row);del h,ids,target,loss,norm
  torch.cuda.synchronize();loop_seconds=time.perf_counter()-loop_start
  assert local_updates==protocol['steps_per_job']==16
  save_checkpoint(out/f'checkpoint-{step}.pt',params,optimizer,scheduler,step,new_identity,dict(rows=rows))
  state['k']=0;model.eval();model.gradient_checkpointing_disable()
  final_cal=[nll(w) for w in cal]
  result=dict(status='complete',job=job,identity=new_identity,environment=environment,started_utc=started,finished_utc=utc(),seconds=time.perf_counter()-tick,step=step,optimizer_updates=local_updates,task_predictions=0,rows=rows,loop_seconds=loop_seconds,complete_step_seconds=sum(x['seconds'] for x in rows),peak_allocated_bytes=torch.cuda.max_memory_allocated(),calibration_values=vals,final_common_dense_calibration_values=final_cal,calibration_max_abs_error=error,checkpoint_sha256=sha(checkpoint),final_checkpoint_sha256=sha(out/f'checkpoint-{step}.pt'),eval_source_sha256=sha(Path(__file__)))

 except Exception:
  result=dict(status='failed',job=job,started_utc=started,finished_utc=utc(),optimizer_updates=local_updates,step=step,error=traceback.format_exc())
 save(out/'result.json',result)
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
