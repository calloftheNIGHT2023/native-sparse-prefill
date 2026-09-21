"""Check shifted causal alignment and suffix equivalence before model evaluation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import json,hashlib,functools,time
from datetime import datetime,timezone
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
import word_suffix_attention as op
R=Path(__file__).resolve().parents[1]
def main():
    out=R/'results/word-suffix-preflight-v1';out.mkdir(exist_ok=False);checks=[]
    torch.set_num_threads(4);torch.manual_seed(2026091698);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    op.base.LOCKED_POOL_CONFIG=(32,4,3);op.base.flash_moba_attn_varlen_func=functools.partial(op.base.flash_moba_attn_varlen_func,deterministic=True)
    with torch.no_grad():
        for n in [8192,32768]:
            q=torch.randn(n,14,64,device='cuda',dtype=torch.bfloat16);k=torch.randn(n,2,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);scale=64**-.5;start=n-16
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):full=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True,dropout_p=0.,scale=scale,enable_gqa=True)[0].transpose(0,1)
            tail=op.dense_suffix(q,k,v,scale,start);err=float((tail.float()-full[start:].float()).abs().max());assert torch.allclose(tail,full[start:],atol=.003,rtol=.03),err
            # Explicit FP32 causal reference verifies every one of the 16 suffix rows.
            qf=q[start:].transpose(0,1).float();kf=k.transpose(0,1).repeat_interleave(7,0).float();vf=v.transpose(0,1).repeat_interleave(7,0).float()
            mask=torch.arange(n,device='cuda')[None,:]<=torch.arange(start,n,device='cuda')[:,None]
            ref=(torch.softmax((qf@kf.transpose(-1,-2)*scale).masked_fill(~mask,float('-inf')),dim=-1)@vf).transpose(0,1)
            referr=float((tail.float()-ref).abs().max());assert torch.allclose(tail.float(),ref,atol=.003,rtol=.03),referr
            sparse=op.base.sparse(q,k,v,128,32,'fp32',scale);hybrid=op.sparse_prefix_dense_suffix(q,k,v,scale,start)
            assert torch.equal(sparse[:start],hybrid[:start]) and torch.equal(hybrid[start:],tail) and torch.isfinite(hybrid).all()
            checks.append(dict(n=n,suffix_tokens=16,dense_suffix_max_abs_error=err,fp32_reference_max_abs_error=referr,prefix_bitwise_unchanged=True))
    result=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),checks=checks,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256((R/'scripts/word_suffix_attention.py').read_bytes()).hexdigest(),optimizer_updates=0,task_predictions=0,scope='Random tensor causal-correctness check only, no task result or speed claim.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
