"""GPU controls for exact sparse metadata reconstruction and forced routing."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,json
from pathlib import Path
from task_route_intervention import routed_attention,replace_question_blocks,metadata_from_ids
torch.manual_seed(2026091607);torch.set_num_threads(4)
checks=[]
for n in [512,2048,4096]:
 q=torch.randn(n,14,64,device='cuda',dtype=torch.bfloat16);k=torch.randn(n,2,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
 native=routed_attention(q,k,v,0.125);rebuilt=routed_attention(q,k,v,0.125,'rebuild')
 error=float((native-rebuilt).abs().max());assert error==0
 checks.append(dict(n=n,rebuild_max_abs_error=error))
 if n==4096:
  y,forced=routed_attention(q,k,v,.125,'forced',start=4000,forced=[3,4],target=[3,4],return_ids=True);assert torch.isfinite(y).all()
  assert torch.equal(y[:4000],native[:4000])
  assert (forced[-1]>=0).sum().item()==14*16 and all(i in forced[-1,0].tolist() for i in [3,4,31])
  refs=[]
  for head in range(14):
   token_ids=(forced[-1,head,:,None].long()*128+torch.arange(128,device=q.device)[None,:]).flatten()
   kk=k[token_ids,head//7].float();vv=v[token_ids,head//7].float()
   refs.append(torch.softmax(q[-1,head].float()@kk.T*.125,dim=-1)@vv)
  ref=torch.stack(refs);err=float((ref-y[-1].float()).abs().max())
  assert torch.allclose(ref,y[-1].float(),atol=.03,rtol=.03)
  checks.append(dict(forced_prefix_unchanged=True,finite=True,budget_and_local_preserved=True,fp32_reference_max_error=err))
out=Path('results/task-route-preflight-v1');out.mkdir(parents=True,exist_ok=False)
(out/'result.json').write_text(json.dumps(dict(status='passed',checks=checks,optimizer_updates=0),indent=2)+'\n');print(json.dumps(checks))
