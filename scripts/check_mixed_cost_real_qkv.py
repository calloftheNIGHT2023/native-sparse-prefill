"""Independent same-QKV output/backward reference at the diagnostic's actual first layer."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
import torch.nn.functional as F
import run_flashmoba_realtext_precision as base
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/flashmoba-mixed-cost-real-qkv-v0';OUT.mkdir(parents=True,exist_ok=False)
(OUT/'source.py').write_bytes(Path(__file__).read_bytes());start=datetime.now(timezone.utc).isoformat()
torch.set_num_threads(4);torch.manual_seed(2026091555);torch.backends.cuda.matmul.allow_tf32=False;base.LOCKED_POOL_CONFIG=(32,4,3)
capture=torch.load(ROOT/'results/flashmoba-mixed-cost-rounding-diagnostic-v0/first-layer-qkv.pt',map_location='cpu',weights_only=True)
same=all(torch.equal(capture['flash'][k],c[k]) for c in capture.values() for k in ['query','key','value']);assert same
q,k,v=[capture['flash'][name].cuda().to(torch.bfloat16).detach().requires_grad_() for name in ['query','key','value']]
qr,kr,vr=[x.detach().float().requires_grad_() for x in [q,k,v]]
n=q.shape[-2];mask=torch.ones(n,n,device='cuda',dtype=torch.bool).tril();scale=q.shape[-1]**-.5
scores=(qr@kr.repeat_interleave(7,dim=1).transpose(-1,-2))*scale
expected=torch.softmax(scores.masked_fill(~mask,float('-inf')),dim=-1)@vr.repeat_interleave(7,dim=1)
go=torch.randn_like(q);gr=torch.autograd.grad(expected,(qr,kr,vr),go.float())
rows=[];stored={'q':q.detach().cpu(),'k':k.detach().cpu(),'v':v.detach().cpu(),'output_gradient':go.cpu(),'reference_output':expected.detach().cpu(),'reference_gradients':[g.cpu() for g in gr]}
def metrics(actual,reference):
 a=actual.float();r=reference.float();bad=~torch.isclose(a,r,atol=.03,rtol=.03)
 return dict(relative_l2=float((a-r).norm()/r.norm()),max_abs=float((a-r).abs().max()),elements=a.numel(),outside_tolerance=int(bad.sum()),passed=not bool(bad.any()))
for mode in ['flash','sparse_allblocks']:
 if mode=='flash':
  with sdpa_kernel(SDPBackend.FLASH_ATTENTION):actual=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale,enable_gqa=True)
 else:actual=base.sparse(q[0].transpose(0,1).contiguous(),k[0].transpose(0,1).contiguous(),v[0].transpose(0,1).contiguous(),128,2,'fp32',scale).transpose(0,1)[None]
 grad=torch.autograd.grad(actual,(q,k,v),go);row=dict(mode=mode,output=metrics(actual,expected),gradients={name:metrics(a,b) for name,a,b in zip(['q','k','v'],grad,gr)})
 row['all_passed']=row['output']['passed'] and all(g['passed'] for g in row['gradients'].values());rows.append(row)
 stored[mode]={'output':actual.detach().cpu(),'gradients':[g.cpu() for g in grad]}
torch.save(stored,OUT/'replay-tensors.pt')
result=dict(status='complete',started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),same_first_layer_qkv_across_modes=same,
            length=n,rows=rows,all_passed=all(r['all_passed'] for r in rows),scientific_optimizer_updates=0,
            atol=.03,rtol=.03,original_full_model_amp_gate='failed_not_waived',
            scope='One actual 256-token first-layer QKV, identical input and external output gradient, independent FP32 causal GQA reference. All-block comparison only, not a 16K or full-training guarantee.')
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
