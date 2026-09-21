"""Locked checkpoint evaluation, calibration replay before opening new windows."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,functools,hashlib,json,math,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from torch.nn.attention import sdpa_kernel,SDPBackend
from run_amp_recovery import LoRALinear
from amp_recovery_config_identity import verify_config_identity
import run_flashmoba_realtext_precision as base
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main(a):
 a.output.mkdir(parents=True,exist_ok=False);started=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
 cfg=read(a.run_dir/'config.json');run=read(a.run_dir/'result.json');index=read(a.run_dir/'checkpoint-index.json')
 protocol=read(R/'provenance/amp-recovery-confirmation-protocol.json')
 assert run['phase']=='train' and run['status']=='complete' and run['step']==256 and index['step']==256
 k,seed,lr=[run['identity'][x] for x in ['k','seed','lr']]
 assert [k,seed,lr] in protocol['locked_conditions']
 assert cfg['deterministic_backward'] and cfg['length']==16384 and cfg['lora_rank']==8 and cfg['lora_alpha']==16
 config_verification=verify_config_identity(a.run_dir)
 assert all(sha(R/'scripts'/n)==h for n,h in run['identity']['sources'].items())
 cp=a.run_dir/index['path'];assert sha(cp)==index['sha256']
 assert sha(Path(base.flash_moba_cuda.__file__))==cfg['extension_sha256']
 torch.use_deterministic_algorithms(True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 base.flash_moba_attn_varlen_func=functools.partial(base.flash_moba_attn_varlen_func,deterministic=True);base.LOCKED_POOL_CONFIG=(32,4,3)
 assert run['environment']==dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()))
 for x in read(R/'data/flashmoba-qwen-precision-v0/manifest.json')['files']:
  if x['path'].startswith('model/'):assert sha(R/'data/flashmoba-qwen-precision-v0'/x['path'])==x['sha256']
 torch.manual_seed(seed)
 model,info=AutoModelForCausalLM.from_pretrained(R/cfg['model_path'],local_files_only=True,trust_remote_code=False,use_safetensors=True,attn_implementation='eager',dtype=torch.float32,output_loading_info=True)
 assert not info['missing_keys'] and not info['unexpected_keys'] and not info.get('mismatched_keys')
 model=model.cuda().eval();model.requires_grad_(False);torch.manual_seed(seed)
 for layer in model.model.layers:
  for n in ['q_proj','k_proj','v_proj','o_proj']:setattr(layer.self_attn,n,LoRALinear(getattr(layer.self_attn,n),8,16))
 params={n:p for n,p in model.named_parameters() if p.requires_grad}
 assert sum(p.numel() for p in params.values())==1081344
 pp=R/'provenance/checkpoint-quality-cost-protocol.json';curve_protocol=read(pp);spec=curve_protocol['runs'][a.name]
 assert [k,seed,lr]==[spec['k'],spec['seed'],spec['lr']]
 assert sha(a.run_dir/'result.json')==spec['training_result_sha256']
 for name,digest in curve_protocol['sources'].items():assert sha(R/name)==digest,name
 def attention(module,query,key,value,attention_mask,scaling=None,dropout=0.,**kw):
  assert query.shape[0]==1 and query.shape[-2]==key.shape[-2] and dropout==0
  q,kk,v=[x[0].transpose(0,1).to(torch.bfloat16).contiguous() for x in (query,key,value)]
  if k==0:
   with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    y=F.scaled_dot_product_attention(q.transpose(0,1)[None],kk.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True,dropout_p=0.,scale=scaling,enable_gqa=True)
   return y.transpose(1,2).to(query.dtype),None
  return base.sparse(q,kk,v,128,k,'fp32',scaling)[None].to(query.dtype),None
 ALL_ATTENTION_FUNCTIONS.register('checkpoint_curve',attention);model.config._attn_implementation='checkpoint_curve'
 @torch.no_grad()
 def nll(w):
  ids=torch.tensor(w[:-1][None],device='cuda');target=torch.tensor(w[1:],device='cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):h=model.model(ids,use_cache=False).last_hidden_state
  total=0.
  for i in range(0,len(target),cfg['chunk_size']):
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=model.lm_head(h[:,i:i+cfg['chunk_size']])
   total+=float(F.cross_entropy(logits[0].float(),target[i:i+cfg['chunk_size']],reduction='sum'))
  value=total/len(target);assert math.isfinite(value);return value
 calfile=R/'data/flashmoba-amp-recovery-v0/train-calibration.npz';assert sha(calfile)==cfg['data_sha256']
 cal=np.load(calfile)['calibration']
 heldfile=R/'data/amp-confirmation-v0/heldout.npz';assert sha(heldfile)==curve_protocol['heldout_sha256']
 held=np.load(heldfile)['heldout'];assert held.shape==(29,16385)
 taskdir=R/'data/task-quality-v0';task_protocol=read(taskdir/'protocol.json')
 for name in ['formal.json','formal.npz']:assert sha(taskdir/name)==task_protocol['files'][name]
 metadata=read(taskdir/'formal.json');arrays=np.load(taskdir/'formal.npz');flat=arrays['input_ids'];offsets=arrays['offsets']
 rows=[(i,x) for i,x in enumerate(metadata) if x['task']=='race_mc' and x['variant'] in curve_protocol['variants']]
 assert len(rows)==192 and len(set(x['item_id'] for _,x in rows))==96
 labels=torch.tensor(task_protocol['label_token_ids'],device='cuda')
 old_name=('dense' if k==0 else 'sparse')+str(seed-2026091560)
 old={ (x['item_id'],x['variant']):x for x in read(R/f'results/task-quality-stage-v0/{old_name}/result.json')['predictions']}
 old_held=read(R/f'results/amp-recovery-stage-v1/heldout-k{k}-seed{seed}/result.json')
 assert old_held['training_result_sha256']==spec['training_result_sha256']
 @torch.no_grad()
 def score(tokens):
  ids=torch.as_tensor(tokens.astype(np.int64)[None],device='cuda')
  torch.cuda.synchronize();st=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   h=model.model(ids,use_cache=False).last_hidden_state[:,-1:];logits=model.lm_head(h)[0,0].float()
  torch.cuda.synchronize();seconds=time.perf_counter()-st
  values=logits[labels].cpu().tolist();return int(np.argmax(values)),values,seconds
 outputs=[]
 for step in curve_protocol['step_order'][str(seed)]:
  cp=a.run_dir/f'checkpoint-{step}.pt';assert sha(cp)==spec['checkpoints'][str(step)]
  state=torch.load(cp,map_location='cpu',weights_only=False)
  assert state['identity']==run['identity'] and state['step']==state['data_cursor']==step and set(params)==set(state['params'])
  assert state['extra']['rows']==run['trace'][:step] and len(state['extra']['rows'])==step
  with torch.no_grad():
   for name,p in params.items():assert torch.isfinite(state['params'][name]).all();p.copy_(state['params'][name])
  del state;model.requires_grad_(False)
  calibration=[nll(w) for w in cal]
  expected=[x for x in run['evaluations'] if x['step']==step and x['split']=='calibration']
  replay_error=None
  if expected:
   assert len(expected)==1;replay_error=max(abs(x-y) for x,y in zip(calibration,expected[0]['values']));assert replay_error<=1e-6
  else:assert step==2
  held_values=[nll(w) for w in held]
  held_error=None
  if step==256:
   held_error=max(abs(x-y) for x,y in zip(held_values,old_held['values']));assert held_error<=1e-6
  predictions=[];control_errors=[]
  for variant in curve_protocol['variants']:
   variant_rows=[(i,x) for i,x in rows if x['variant']==variant];assert len(variant_rows)==96
   for j,(i,row) in enumerate(variant_rows):
    if j==0:
     for _ in range(2):score(flat[offsets[i]:offsets[i+1]])
    guess,values,seconds=score(flat[offsets[i]:offsets[i+1]])
    if step==256:
     p=old[row['item_id'],variant];error=max(abs(x-y) for x,y in zip(values,p['choice_logits']))
     assert guess==p['prediction'] and error<=1e-6;control_errors.append(error)
    entry=dict(step=step,item_id=row['item_id'],variant=variant,gold=row['gold'],prediction=guess,correct=guess==row['gold'],choice_logits=values,diagnostic_forward_seconds=seconds)
    predictions.append(entry)
    with (a.output/'predictions.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
  row=dict(step=step,checkpoint_sha256=spec['checkpoints'][str(step)],training_seconds=sum(x['seconds'] for x in run['trace'][:step]),
           training_target_exposures=step*16384,calibration_values=calibration,calibration_replay_max_abs_error=replay_error,
           heldout_values=held_values,heldout_mean_nll=float(np.mean(held_values)),heldout_256_replay_error=held_error,
           task_256_max_logit_error=max(control_errors) if control_errors else None,predictions=predictions,finished_utc=datetime.now(timezone.utc).isoformat())
  outputs.append(row);(a.output/f'step-{step}.json').write_text(json.dumps(row,indent=2)+'\n')
  print(json.dumps(dict(step=step,training_seconds=row['training_seconds'],heldout_nll=row['heldout_mean_nll'],correct={v:sum(x['correct'] for x in predictions if x['variant']==v) for v in curve_protocol['variants']})),flush=True)
 result=dict(status='complete',name=a.name,k=k,seed=seed,lr=lr,optimizer_updates=0,protocol_sha256=sha(pp),
             training_result_sha256=spec['training_result_sha256'],task_protocol_sha256=sha(taskdir/'protocol.json'),steps=outputs,
             task_predictions=sum(len(x['predictions']) for x in outputs),nll_window_forwards=len(outputs)*(len(cal)+len(held)),
             started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-tick)
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
