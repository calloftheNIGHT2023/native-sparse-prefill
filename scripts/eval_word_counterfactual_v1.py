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
 protocol=json.loads((R/'provenance/word-counterfactual-protocol-v1.json').read_text());job=next(x for x in protocol['jobs'] if x['name']==a.job)
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
  data=R/'data/word-counterfactual-v0';meta=json.loads((data/'tasks.json').read_text());a=np.load(data/'tasks.npz');flat=a['input_ids'];offsets=a['offsets']
  labels=torch.tensor(protocol['answer_token_ids'],device='cuda');predictions=[];short_gate=None;long_gate=None
  def metrics(variant):
   xs=[x for x in predictions if x['variant']==variant];groups={}
   for x in xs:groups.setdefault(x['family_id'],[]).append(x)
   assert len(xs)==64 and len(groups)==32 and all(len(x)==2 for x in groups.values())
   return dict(correct=sum(x['correct'] for x in xs),total=64,pairs_both_correct=sum(all(y['correct'] for y in x) for x in groups.values()),pairs_total=32)
  def gate(variant,criteria):
   m=metrics(variant);n=metrics('no_context');m['gain_over_no_context']=m['correct']-n['correct'];m['passed']=m['correct']>=criteria['min_correct'] and m['pairs_both_correct']>=criteria['min_pairs_both_correct'] and m['gain_over_no_context']>=criteria['min_gain_over_no_context'];return m
  with torch.no_grad():
   for j,item in enumerate(meta):
    if item['variant']=='long32768' and short_gate is None:
     short_gate=gate('short',protocol['short_gate']);event('short_ability_gate',**short_gate)
    ids=torch.tensor(flat[offsets[j]:offsets[j+1]].astype(np.int64)[None],device='cuda')
    with amp():h=model.model(ids,use_cache=False).last_hidden_state[:,-1:];logits=model.lm_head(h)[0,0].float()
    values=logits[labels].cpu().tolist();assert np.isfinite(values).all();guess=int(np.argmax(values))
    pred=dict(**item,prediction=guess,correct=guess==item['gold'],choice_logits=values);predictions.append(pred)
    with (out/'task-predictions.jsonl').open('a') as f:f.write(json.dumps(pred)+'\n')
    if len(predictions)%16==0:event('prediction_progress',completed=len(predictions),maximum=192)
  long_gate=gate('long32768',protocol['long_gate']);event('long_ability_gate',**long_gate)
  assert len(predictions)==192
  result=dict(status='complete',job=job,identity=identity,environment=environment,started_utc=started,finished_utc=utc(),seconds=time.perf_counter()-tick,step=step,optimizer_updates=0,task_predictions=len(predictions),short_gate=short_gate,long_gate=long_gate,no_context=metrics('no_context'),calibration_values=vals,calibration_max_abs_error=error,checkpoint_sha256=sha(checkpoint),eval_source_sha256=sha(Path(__file__)),predictions=predictions)
 except Exception:
  result=dict(status='failed',job=job,started_utc=started,finished_utc=utc(),optimizer_updates=0,error=traceback.format_exc())
 save(out/'result.json',result)
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
