"""Diagnose, without waiving, the failed AMP gate using matched independent arithmetic."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import inspect,json,hashlib,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
from run_pool_training_control import LoRALinear
import run_flashmoba_realtext_precision as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-mixed-cost-rounding-diagnostic-v0';OUT.mkdir(parents=True,exist_ok=False)
start=datetime.now(timezone.utc).isoformat();(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
(OUT/'native-eager-source.txt').write_text(inspect.getsource(eager_attention_forward))
cfg=json.loads((ROOT/'configs/flashmoba-mixed-cost-v0.json').read_text())
assert hashlib.sha256(Path(base.flash_moba_cuda.__file__).read_bytes()).hexdigest()==cfg['candidate_extension_sha256']
torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
base.LOCKED_POOL_CONFIG=(32,4,3)
data=np.load(ROOT/'data/flashmoba-quality-cost-v0/tokens.npz')['train'][0]
ids=torch.tensor(data[:256][None],device='cuda');target=torch.tensor(data[1:257],device='cuda')
model=AutoModelForCausalLM.from_pretrained(ROOT/'data/flashmoba-qwen-precision-v0/model',local_files_only=True,
    trust_remote_code=False,use_safetensors=True,attn_implementation='eager',dtype=torch.float32).cuda().eval();model.requires_grad_(False)
torch.manual_seed(cfg['seed'])
for layer in model.model.layers:
    for name in ['q_proj','k_proj','v_proj','o_proj']:
        setattr(layer.self_attn,name,LoRALinear(getattr(layer.self_attn,name),8,16))
params=[p for p in model.parameters() if p.requires_grad]
def digest():return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in params)).hexdigest()
initial=digest();state=dict(mode='native');captures={}

def attn(module,q,k,v,attention_mask,scaling=None,dropout=0.,**kwargs):
    mode=state['mode'];n=q.shape[-2]
    if module.layer_idx==0 and mode not in captures:captures[mode]=dict(query=q.detach().cpu(),key=k.detach().cpu(),value=v.detach().cpu())
    causal=torch.ones(n,n,device=q.device,dtype=torch.bool).tril()
    if mode=='eager_clone':
        mask=torch.zeros(1,1,n,n,device=q.device,dtype=torch.float32).masked_fill(~causal,torch.finfo(torch.float32).min)
        return eager_attention_forward(module,q,k,v,mask,scaling,dropout=0.)
    qb,kb,vb=[x.to(torch.bfloat16) for x in (q,k,v)]
    if mode=='flash':
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y=F.scaled_dot_product_attention(qb,kb,vb,is_causal=True,scale=scaling,dropout_p=0.,enable_gqa=True)
    elif mode=='sparse_allblocks':
        y=base.sparse(qb[0].transpose(0,1).contiguous(),kb[0].transpose(0,1).contiguous(),vb[0].transpose(0,1).contiguous(),128,2,'fp32',scaling)
        return y[None].to(q.dtype),None
    else:
        with torch.autocast('cuda',enabled=False):
            qf=qb.float();kf=kb.float().repeat_interleave(q.shape[1]//k.shape[1],dim=1);vf=vb.float().repeat_interleave(q.shape[1]//v.shape[1],dim=1)
            scores=qf@kf.transpose(-2,-1)
            if mode=='round_scores_probs':
                scores=scores.to(torch.bfloat16)
                scores=(scores*scaling).to(torch.bfloat16).float()
            else:scores=scores*scaling
            probs=torch.softmax(scores.masked_fill(~causal,float('-inf')),dim=-1)
            if mode in ['round_probs','round_scores_probs']:probs=probs.to(torch.bfloat16).float()
            y=(probs@vf).to(torch.bfloat16)
    return y.transpose(1,2).to(q.dtype),None

ALL_ATTENTION_FUNCTIONS.register('rounding_diagnostic',attn)
values={};rows=[]
for mode in ['native','eager_clone','math_fp32','round_probs','round_scores_probs','flash','sparse_allblocks']:
    state['mode']=mode;model.config._attn_implementation='eager' if mode=='native' else 'rounding_diagnostic';model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(ids,use_cache=False).logits;loss=F.cross_entropy(logits[0].float(),target)
    loss.backward();grad=torch.cat([p.grad.detach().float().flatten() for p in params])
    assert bool(torch.isfinite(grad).all())
    values[mode]=dict(logits=logits.detach().float().cpu(),gradient=grad.cpu(),nll=float(loss.detach()))
    rows.append(dict(mode=mode,nll=values[mode]['nll'],gradient_norm=float(grad.norm()),utc=datetime.now(timezone.utc).isoformat()))
    del logits,loss,grad

def compare(left,right):
    a,b=values[left],values[right]
    return dict(reference=left,compared=right,nll_difference=b['nll']-a['nll'],
        logits_relative_l2=float((b['logits']-a['logits']).norm()/a['logits'].norm()),
        gradient_relative_l2=float((b['gradient']-a['gradient']).norm()/a['gradient'].norm()),
        gradient_cosine=float(F.cosine_similarity(a['gradient'],b['gradient'],dim=0)),
        logits_bitwise_equal=torch.equal(a['logits'],b['logits']),gradient_bitwise_equal=torch.equal(a['gradient'],b['gradient']))
pairs=[compare('native',m) for m in values if m!='native']+[compare('math_fp32',m) for m in ['round_probs','round_scores_probs','flash','sparse_allblocks']]+[compare('flash','sparse_allblocks')]
state['mode']='flash';model.config._attn_implementation='rounding_diagnostic'
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):sample=model(ids,use_cache=True,logits_to_keep=1)
cache=dict(logits_shape=list(sample.logits.shape),cache_sequence_length=sample.past_key_values.get_seq_length());del sample
assert cache['logits_shape'][1]==1 and cache['cache_sequence_length']==256
assert digest()==initial
torch.save({m:dict(gradient=x['gradient'],last_token_logits=x['logits'][:,-1].clone(),
    full_logits_sha256=hashlib.sha256(x['logits'].numpy().tobytes()).hexdigest(),nll=x['nll']) for m,x in values.items()},OUT/'comparison-gradients.pt')
torch.save(captures,OUT/'first-layer-qkv.pt')
result=dict(status='complete',started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),rows=rows,pairs=pairs,
    cache_api_check=cache,initial_adapters_sha256=initial,adapter_parameters_unchanged=True,scientific_optimizer_updates=0,
    original_amp_gate_status='failed_and_not_waived',
    scope='Post-failure arithmetic diagnosis at 256 tokens on the same training prefix. FP32 math uses identical BF16 QKV quantization and BF16 output as Flash. Rounding variants emulate probability or score+probability materialization; no new quality experiment, no long-context correctness claim, no relaxed original thresholds.')
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
