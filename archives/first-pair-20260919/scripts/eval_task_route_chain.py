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
from task_route_intervention import routed_attention
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
 state=torch.load(cp,map_location='cpu',weights_only=False)
 assert state['identity']==run['identity'] and state['step']==state['data_cursor']==256 and set(params)==set(state['params'])
 with torch.no_grad():
  for n,p in params.items():assert torch.isfinite(state['params'][n]).all();p.copy_(state['params'][n])
 del state;model.requires_grad_(False)
 route_mode='calibration';suffix_start=None;forced=None;target_blocks=None;telemetry=[]
 def attention(module,query,key,value,attention_mask,scaling=None,dropout=0.,**kw):
  assert query.shape[0]==1 and query.shape[-2]==key.shape[-2] and dropout==0
  q,kk,v=[x[0].transpose(0,1).to(torch.bfloat16).contiguous() for x in (query,key,value)]
  if k==0:
   with sdpa_kernel(SDPBackend.FLASH_ATTENTION):y=F.scaled_dot_product_attention(q.transpose(0,1)[None],kk.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True,dropout_p=0.,scale=scaling,enable_gqa=True)
   return y.transpose(1,2).to(query.dtype),None
  if route_mode=='calibration':return base.sparse(q,kk,v,128,k,'fp32',scaling)[None].to(query.dtype),None
  out=routed_attention(q,kk,v,scaling,mode='rebuild' if route_mode=='baseline' else 'forced',start=suffix_start,forced=forced,target=target_blocks,telemetry=telemetry)
  return out[None].to(query.dtype),None
 ALL_ATTENTION_FUNCTIONS.register('confirmation',attention);model.config._attn_implementation='confirmation'
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
 cal=np.load(calfile)['calibration'];replay=[nll(w) for w in cal];expected=run['evaluations'][-1]['values']
 replay_error=max(abs(x-y) for x,y in zip(replay,expected));assert replay_error<=1e-6,(replay_error,replay,expected)
 assert k==16 and read(R/'results/task-route-preflight-v1/result.json')['status']=='passed'
 route_protocol=read(R/'provenance/task-quality-route-chain-protocol.json')
 assert sha(R/'scripts/task_route_intervention.py')==route_protocol['route_source_sha256']
 taskdir=R/'data/task-quality-v0';task_protocol=read(taskdir/'protocol.json')
 for name in ['formal.json','formal.npz']:assert sha(taskdir/name)==task_protocol['files'][name]
 metadata=read(taskdir/'formal.json');arrays=np.load(taskdir/'formal.npz');flat=arrays['input_ids'];offsets=arrays['offsets']
 labels=torch.tensor(task_protocol['label_token_ids'],device='cuda')
 chosen=route_protocol['screen_ids' if a.part=='screen' else 'remaining_ids']
 rows=[(i,x) for i,x in enumerate(metadata) if x['task']=='race_mc' and x['variant']=='long16384' and x['item_id'] in chosen]
 assert len(rows)==len(chosen)
 placements={x['item_id']:x for x in route_protocol['rows']}
 old={x['item_id']:x for x in read(R/f'results/task-quality-stage-v0/sparse{seed-2026091560}/result.json')['predictions'] if x['variant']=='long16384'}
 @torch.no_grad()
 def score(tokens):
  ids=torch.as_tensor(tokens.astype(np.int64)[None],device='cuda')
  torch.cuda.synchronize();st=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   h=model.model(ids,use_cache=False).last_hidden_state[:,-1:];logits=model.lm_head(h)[0,0].float()
  torch.cuda.synchronize();seconds=time.perf_counter()-st
  values=logits[labels].cpu().tolist();return int(np.argmax(values)),values,seconds
 predictions=[];control_errors=[]
 for condition in route_protocol['order'][str(seed)]:
  route_mode=condition
  for j,(i,row) in enumerate(rows):
   placement=placements[row['item_id']];suffix_start=placement['suffix_start'];target_blocks=placement['target_blocks'];forced=placement['sham_blocks'] if condition=='sham' else target_blocks
   if j==0:
    for _ in range(2):score(flat[offsets[i]:offsets[i+1]])
   telemetry.clear();guess,values,seconds=score(flat[offsets[i]:offsets[i+1]])
   assert len(telemetry)==24
   if condition=='baseline':
    expected=old[row['item_id']];error=max(abs(x-y) for x,y in zip(values,expected['choice_logits']))
    assert guess==expected['prediction'] and error<=1e-6,(row['item_id'],error);control_errors.append(error)
   entry=dict(condition=condition,item_id=row['item_id'],gold=row['gold'],prediction=guess,correct=guess==row['gold'],choice_logits=values,prefill_seconds=seconds,layer_telemetry=list(telemetry),suffix_start=suffix_start,target_blocks=target_blocks,forced_blocks=forced if condition!='baseline' else [])
   predictions.append(entry)
   with (a.output/'predictions.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
  print(json.dumps(dict(condition=condition,correct=sum(x['correct'] for x in predictions if x['condition']==condition),total=len(rows))),flush=True)
 result=dict(status='complete',part=a.part,seed=seed,k=k,lr=lr,optimizer_updates=0,checkpoint_sha256=index['sha256'],training_result_sha256=sha(a.run_dir/'result.json'),calibration_replay_max_abs_error=replay_error,control_max_logit_error=max(control_errors),route_protocol_sha256=sha(R/'provenance/task-quality-route-chain-protocol.json'),task_protocol_sha256=sha(taskdir/'protocol.json'),predictions=predictions,started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-tick)
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--part',choices=['screen','remaining'],required=True);main(p.parse_args())
