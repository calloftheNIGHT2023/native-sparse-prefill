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
from chunked_lm_loss import chunked_head_backward
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
 def attention(module,query,key,value,attention_mask,scaling=None,dropout=0.,**kw):
  assert query.shape[0]==1 and query.shape[-2]==key.shape[-2] and dropout==0
  q,kk,v=[x[0].transpose(0,1).to(torch.bfloat16).contiguous() for x in (query,key,value)]
  if k==0:
   with sdpa_kernel(SDPBackend.FLASH_ATTENTION):y=F.scaled_dot_product_attention(q.transpose(0,1)[None],kk.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True,dropout_p=0.,scale=scaling,enable_gqa=True)
   return y.transpose(1,2).to(query.dtype),None
  return base.sparse(q,kk,v,128,k,'fp32',scaling)[None].to(query.dtype),None
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
 assert k==0 and seed==2026091560 and lr==0.0003
 cost_protocol=read(R/'provenance/task-quality-density-cost-v1-protocol.json')
 train=np.load(calfile)['train'];rows=[]
 for p in params.values():p.requires_grad_(True)
 assert sum(p.numel() for p in model.parameters() if p.requires_grad)==1081344
 def param_digest():return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in params.values())).hexdigest()
 initial=param_digest()
 def amp():return torch.autocast('cuda',dtype=torch.bfloat16)
 def fb(w):
  model.zero_grad(set_to_none=True)
  ids=torch.tensor(w[:-1][None],device='cuda');target=torch.tensor(w[1:],device='cuda')
  with amp():h=model.model(ids,use_cache=False).last_hidden_state
  loss=chunked_head_backward(h,model.lm_head,target,cfg['chunk_size'],amp)
  norm=torch.nn.utils.clip_grad_norm_(list(params.values()),cfg['gradient_clip'],error_if_nonfinite=True)
  assert bool(torch.isfinite(loss))
  return float(loss),float(norm)
 for round_index,order in enumerate(cost_protocol['orders']):
  w=train[cost_protocol['windows'][round_index]]
  for mode in order:
   k=mode
   for _ in range(cost_protocol['warmups_per_cell']):fb(w)
   times=[];losses=[];norms=[];torch.cuda.reset_peak_memory_stats()
   for _ in range(cost_protocol['samples_per_cell']):
    torch.cuda.synchronize();st=time.perf_counter();loss,norm=fb(w);torch.cuda.synchronize()
    times.append(time.perf_counter()-st);losses.append(loss);norms.append(norm)
   row=dict(round=round_index,k=mode,window=cost_protocol['windows'][round_index],times=times,losses=losses,gradient_norms=norms,median_seconds=float(np.median(times)),peak_bytes=torch.cuda.max_memory_allocated(),utc=datetime.now(timezone.utc).isoformat())
   rows.append(row)
   with (a.output/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
 assert initial==param_digest()
 candidates=[]
 for mode in [16,32]:
  ratios=[next(x['median_seconds'] for x in rows if x['k']==mode and x['round']==r)/next(x['median_seconds'] for x in rows if x['k']==0 and x['round']==r) for r in [0,1]]
  candidates.append(dict(k=mode,ratios=ratios,both_rounds_at_least_5pct_faster=all(v<=0.95 for v in ratios)))
 result=dict(status='complete',started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-tick,rows=rows,candidates=candidates,optimizer_updates=0,weights_unchanged=True,initial_params_sha256=initial,checkpoint_sha256=index['sha256'],calibration_replay_max_abs_error=replay_error,protocol_sha256=sha(R/'provenance/task-quality-density-cost-v1-protocol.json'),scope=cost_protocol['scope'])
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(candidates),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
