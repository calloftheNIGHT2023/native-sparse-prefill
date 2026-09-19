"""Bounded K32 route-reconstruction and oracle intervention implementation checks."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from datetime import datetime,timezone
import torch,json,time,hashlib,functools
import run_flashmoba_realtext_precision as base
from word_route_intervention import routed_attention
R=Path(__file__).resolve().parents[1]
out=R/'results/word-route-preflight-v0';out.mkdir(exist_ok=False)
torch.manual_seed(2026091696);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
base.LOCKED_POOL_CONFIG=(32,4,3);base.flash_moba_attn_varlen_func=functools.partial(base.flash_moba_attn_varlen_func,deterministic=True)
started=datetime.now(timezone.utc).isoformat();checks=[]
with torch.no_grad():
 for n in [8192,32768]:
  q=torch.randn(n,14,64,device='cuda',dtype=torch.bfloat16);k=torch.randn(n,2,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
  native=base.sparse(q,k,v,128,32,'fp32',.125);rebuilt=routed_attention(q,k,v,.125,'rebuild')
  err=float((native-rebuilt).abs().max());assert err==0
  start=n-32;target=[n//512];times=[]
  for _ in range(3):
   torch.cuda.synchronize();tick=time.perf_counter();y,ids=routed_attention(q,k,v,.125,'forced',start=start,forced=target,target=target,return_ids=True);torch.cuda.synchronize();times.append(time.perf_counter()-tick)
  assert torch.isfinite(y).all() and torch.equal(y[:start],native[:start])
  assert (ids[-1]>=0).sum().item()==14*32 and all(target[0] in ids[-1,h].tolist() and n//128-1 in ids[-1,h].tolist() for h in range(14))
  ref=[]
  for h in range(14):
   token_ids=(ids[-1,h,:,None].long()*128+torch.arange(128,device='cuda')[None,:]).flatten()
   kk=k[token_ids,h//7].float();vv=v[token_ids,h//7].float();ref.append(torch.softmax(q[-1,h].float()@kk.T*.125,dim=-1)@vv)
  ref=torch.stack(ref);torch.testing.assert_close(ref,y[-1].float(),atol=.03,rtol=.03)
  checks.append(dict(n=n,rebuild_max_abs_error=err,prefix_bitwise_unchanged=True,budget_per_query_head=32,forced_and_local_present=True,fp32_reference_max_error=float((ref-y[-1].float()).abs().max()),forced_layer_seconds=times))
  del q,k,v,native,rebuilt,y,ids,ref
result=dict(status='passed',started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),checks=checks,optimizer_updates=0,task_predictions=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),route_sha256=hashlib.sha256((R/'scripts/word_route_intervention.py').read_bytes()).hexdigest(),scope='Random-tensor mathematical correctness andtiming screen, nottask quality orend-to-end speedup.')
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
