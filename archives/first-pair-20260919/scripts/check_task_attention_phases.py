"""Causal and numerical checks for query-position phase interventions."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,functools
from pathlib import Path
import torch
import run_flashmoba_realtext_precision as base
from task_attention_phases import phase_attention,dense_attention
torch.manual_seed(2026091623);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
base.LOCKED_POOL_CONFIG=(32,4,3)
base.flash_moba_attn_varlen_func=functools.partial(base.flash_moba_attn_varlen_func,deterministic=True)
n,start=4096,4000
q=torch.randn(n,14,64,device='cuda',dtype=torch.bfloat16)
k=torch.randn(n,2,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
s=phase_attention(q,k,v,.125,'SS',start);d=phase_attention(q,k,v,.125,'DD',start)
checks=[]
for c,span,a,b in [('DS',None,0,start),('SD',None,start,n),('target_D',(511,899),511,899),('sham_D',(2200,2588),2200,2588)]:
    y=phase_attention(q,k,v,.125,c,start,span)
    assert torch.equal(y[a:b],d[a:b]) and torch.equal(y[:a],s[:a]) and torch.equal(y[b:],s[b:])
    # A future K/V perturbation cannot affect any earlier output, including in hybrids.
    k2=k.clone();v2=v.clone();k2[3000:]+=3;v2[3000:]-=2
    other=phase_attention(q,k2,v2,.125,c,start,span)
    assert torch.equal(y[:3000],other[:3000])
    checks.append(dict(condition=c,span=[a,b],path_selection_exact=True,future_kv_invariant=True))
refs=[]
for h in range(14):
    p=torch.softmax(q[-1,h].float()@k[:,h//7].float().T*.125,-1)
    refs.append(p@v[:,h//7].float())
ref=torch.stack(refs);err=float((ref-d[-1].float()).abs().max())
assert torch.allclose(ref,d[-1].float(),atol=.03,rtol=.03)
result=dict(status='passed',checks=checks,dense_fp32_reference_max_abs_error=err,optimizer_updates=0)
out=Path('results/task-attention-phases-preflight-v0');out.mkdir(parents=True,exist_ok=False)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
