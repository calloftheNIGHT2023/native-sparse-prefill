"""Fixed zero-update cost pilot; independent of the observed quality report windows."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import contextlib
import hashlib
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from run_pool_training_control import LoRALinear
import run_flashmoba_realtext_precision as base
from chunked_lm_loss import chunked_head_backward

ROOT = Path(__file__).resolve().parents[1]
def utc(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x): p.write_text(json.dumps(x, indent=2)+'\n')

def main(a):
    out = ROOT/'results'/f'flashmoba-long-cost-{a.kind}-{a.length}-v0'
    out.mkdir(parents=True, exist_ok=False)
    started=utc();wall_start=time.perf_counter();rows=[];gates=[]
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    cfg=json.loads((ROOT/'configs/flashmoba-long-cost-v0.json').read_text())
    save(out/'frozen-config.json',cfg)
    def event(kind, **kw):
        row=dict(utc=utc(),event=kind,**kw)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        assert sha(Path(base.flash_moba_cuda.__file__))==cfg['candidate_extension_sha256']
        assert sha(ROOT/'data/flashmoba-quality-cost-v0/tokens.npz')==cfg['tokens_sha256']
        for r in json.loads((ROOT/'data/flashmoba-qwen-precision-v0/manifest.json').read_text())['files']:
            if r['path'].startswith('model/'):
                assert sha(ROOT/'data/flashmoba-qwen-precision-v0'/r['path'])==r['sha256']
        assert json.loads((ROOT/'results/flashmoba-quality-cost-k16-gate-v0/result.json').read_text())['status']=='passed'
        assert json.loads((ROOT/'results/flashmoba-long-backward-barrier-v0/result.json').read_text())['deterministic_repeat_pass']
        torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
        base.LOCKED_POOL_CONFIG=(32,4,3)
        train=np.load(ROOT/'data/flashmoba-quality-cost-v0/tokens.npz')['train']
        stream=np.concatenate([train[0,:-1],train[1,:-1],train[2,:-1],train[3]])
        ids=torch.tensor(stream[:a.length][None],device='cuda')
        targets=torch.tensor(stream[1:a.length+1],device='cuda')
        assert ids.shape==(1,a.length) and targets.shape==(a.length,)
        model=AutoModelForCausalLM.from_pretrained(ROOT/'data/flashmoba-qwen-precision-v0/model',
            local_files_only=True,trust_remote_code=False,use_safetensors=True,
            attn_implementation='eager',dtype=torch.float32).cuda().eval()
        model.requires_grad_(False);torch.manual_seed(cfg['seed'])
        for layer in model.model.layers:
            for name in ['q_proj','k_proj','v_proj','o_proj']:
                setattr(layer.self_attn,name,LoRALinear(getattr(layer.self_attn,name),8,16))
        params=[p for p in model.parameters() if p.requires_grad]
        def param_digest():
            return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in params)).hexdigest()
        initial=param_digest()
        assert sum(p.numel() for p in params)==1081344
        def amp():
            return torch.autocast('cuda',dtype=torch.bfloat16) if a.precision=='amp_bf16' else contextlib.nullcontext()
        state=dict(k=0, math_reference=False)
        def attention(module, query, key, value, attention_mask, scaling=None, dropout=0., **kwargs):
            assert query.shape[0]==1 and query.shape[-2]==key.shape[-2] and dropout==0
            q,k,v=[x[0].transpose(0,1).to(torch.bfloat16).contiguous() for x in (query,key,value)]
            if state['math_reference']:
                with torch.autocast('cuda',enabled=False):
                    qf=q.transpose(0,1)[None].float()
                    kf=k.transpose(0,1)[None].float().repeat_interleave(q.shape[1]//k.shape[1],dim=1)
                    vf=v.transpose(0,1)[None].float().repeat_interleave(q.shape[1]//v.shape[1],dim=1)
                    causal=torch.ones(q.shape[0],k.shape[0],device=q.device,dtype=torch.bool).tril()
                    weights=torch.softmax((qf@kf.transpose(-1,-2)*scaling).masked_fill(~causal,float('-inf')),dim=-1)
                    y=(weights@vf).to(torch.bfloat16)
                return y.transpose(1,2).to(query.dtype),None
            if state['k']==0:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    y=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],
                        is_causal=True,dropout_p=0.,scale=scaling,enable_gqa=True)
                return y.transpose(1,2).to(query.dtype),None
            y=base.sparse(q,k,v,128,state['k'],'fp32',scaling)
            return y[None].to(query.dtype),None
        ALL_ATTENTION_FUNCTIONS.register('mixed_cost',attention)
        # Same-precision native eager vs adapted dense and all-block sparse, including LoRA gradients.
        def gate_eval(implementation,k):
            model.config._attn_implementation=implementation;state['k']=k;model.zero_grad(set_to_none=True)
            with amp():
                logits=model(ids[:,:256],use_cache=False).logits
                loss=F.cross_entropy(logits[0].float(),targets[:256])
            loss.backward()
            grads=torch.cat([p.grad.detach().float().flatten() for p in params])
            assert bool(torch.isfinite(grads).all())
            return logits.detach().float(),float(loss.detach()),grads
        native,nloss,ng=gate_eval('eager',0)
        state['math_reference']=True
        reference,rloss,rg=gate_eval('mixed_cost',0)
        state['math_reference']=False
        dense,dloss,dg=gate_eval('mixed_cost',0)
        sparse,sloss,sg=gate_eval('mixed_cost',2)
        save(out/'original-eager-comparison.json',dict(
            status='diagnostic_original_v0_gate_not_waived',
            logits_relative_l2=float((dense-native).norm()/native.norm()),
            gradient_relative_l2=float((dg-ng).norm()/ng.norm()),nll_abs_difference=abs(dloss-nloss)))
        for name,x,loss,grad in [('matched_math_vs_dense',dense,dloss,dg),('matched_math_vs_allblocks',sparse,sloss,sg)]:
            metrics=dict(name=name,logits_relative_l2=float((x-reference).norm()/reference.norm()),
                         nll_abs_difference=abs(loss-rloss),gradient_relative_l2=float((grad-rg).norm()/rg.norm()),
                         logits_tolerance=.02,nll_tolerance=.05,gradient_tolerance=.10)
            metrics['passed']=metrics['logits_relative_l2']<.02 and metrics['nll_abs_difference']<.05 and metrics['gradient_relative_l2']<.10
            gates.append(metrics);save(out/'gates.json',gates);assert metrics['passed'],metrics
        assert json.loads((ROOT/'results/flashmoba-mixed-cost-real-qkv-v0/result.json').read_text())['all_passed']
        del native,ng,dense,dg,sparse,sg,reference,rg,x,grad
        model.config._attn_implementation='mixed_cost';model.zero_grad(set_to_none=True)
        # A future-token perturbation must leave the first 128 logits unchanged at the actual length.
        # This uses the full model with the strongest K4 mask, no KV cache and no optimizer updates.
        state['k']=4
        with torch.no_grad(),amp():
            prefix=model.model(ids,use_cache=False).last_hidden_state[:,:128].clone()
            altered=ids.clone();altered[:,a.length//2:]=torch.roll(altered[:,a.length//2:],17,1)
            changed=model.model(altered,use_cache=False).last_hidden_state[:,:128]
        delta=float((prefix-changed).abs().max())
        gates.append(dict(name='actual_length_future_perturbation_hidden_prefix',length=a.length,max_abs=delta,passed=delta==0))
        save(out/'gates.json',gates);assert delta==0,delta
        del prefix,altered,changed
        save(out/'manifest.json',dict(started_utc=started,precision=a.precision,length=a.length,
            torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),source_sha256=sha(Path(__file__)),
            helper_source_sha256=sha(Path(base.__file__)),lora_source_sha256=sha(ROOT/'scripts/run_pool_training_control.py'),
            config_sha256=sha(ROOT/'configs/flashmoba-long-cost-v0.json'),initial_adapters_sha256=initial,
            input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            extension_sha256=sha(Path(base.flash_moba_cuda.__file__))))
        event('gates_passed',precision=a.precision,length=a.length,gates=gates)
        def call(mode):
            if mode=='prefill':
                with torch.no_grad(),amp():
                    output=model(ids,use_cache=True,logits_to_keep=1)
                return output
            model.zero_grad(set_to_none=True)
            if mode=='chunked':
                with amp():hidden=model.model(ids,use_cache=False).last_hidden_state
                return chunked_head_backward(hidden,model.lm_head,targets,cfg['chunk_size'],amp)
            with amp():
                logits=model(ids,use_cache=False).logits
                loss=F.cross_entropy(logits[0].float(),targets)
            loss.backward()
            return loss.detach()
        if a.kind=='chunked':
            assert json.loads((ROOT/'results/flashmoba-long-cost-cpu-gate-v0/result.json').read_text())['status']=='passed'
            full_ids,full_targets=ids,targets;ids=ids[:,:2048];targets=targets[:2048]
            for k_gate in [0,4,16]:
                state['k']=k_gate;reference_loss=call('full');reference_grad=torch.cat([p.grad.detach().float().flatten() for p in params])
                chunk_loss=call('chunked');chunk_grad=torch.cat([p.grad.detach().float().flatten() for p in params])
                metrics=dict(name='full_vs_chunked_loss',length=2048,k=k_gate,chunk_size=cfg['chunk_size'],
                    loss_abs_difference=float((chunk_loss-reference_loss).abs()),
                    gradient_relative_l2=float((chunk_grad-reference_grad).norm()/reference_grad.norm()),
                    loss_tolerance=.001,gradient_tolerance=.02)
                metrics['passed']=metrics['loss_abs_difference']<.001 and metrics['gradient_relative_l2']<.02
                gates.append(metrics);save(out/'gates.json',gates);event('chunk_gate',**metrics);assert metrics['passed'],metrics
                del reference_loss,reference_grad,chunk_loss,chunk_grad
            ids,targets=full_ids,full_targets
        for rnd,order in enumerate([[0,4,16],[16,4,0]],1):
            for k in order:
                state['k']=k
                for mode in [a.kind]:
                    model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
                    block_started=utc()
                    for _ in range(cfg['warmups']):
                        output=call(mode);del output
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();samples=[]
                    for i in range(cfg['samples']):
                        torch.cuda.synchronize();t=time.perf_counter();b=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
                        b.record();output=call(mode);e.record();torch.cuda.synchronize()
                        wall=time.perf_counter()-t;gpu_ms=b.elapsed_time(e)
                        samples.append(dict(index=i,utc=utc(),wall_seconds=wall,gpu_span_ms=gpu_ms));del output
                    if mode=='chunked':
                        grads=torch.cat([p.grad.detach().float().flatten() for p in params]);assert bool(torch.isfinite(grads).all())
                        gradnorm=float(grads.norm());del grads
                    else:gradnorm=None
                    peak=torch.cuda.max_memory_allocated()
                    row=dict(precision=a.precision,length=a.length,k=k,round=rnd,mode=mode,started_utc=block_started,finished_utc=utc(),
                             median_wall_seconds=float(np.median([s['wall_seconds'] for s in samples])),
                             wall_iqr_seconds=np.quantile([s['wall_seconds'] for s in samples],[.25,.75]).tolist(),
                             peak_gpu_bytes=peak,gradient_norm=gradnorm,samples=samples)
                    rows.append(row);save(out/'partial.json',rows)
                    event('block_complete',**{key:value for key,value in row.items() if key!='samples'})
        assert param_digest()==initial,'Parameters changed despite zero-update protocol.'
        result=dict(status='complete',started_utc=started,finished_utc=utc(),wall_seconds=time.perf_counter()-wall_start,
                    precision=a.precision,length=a.length,kind=a.kind,rows=rows,gates=gates,scientific_optimizer_updates=0,
                    adapter_parameters_unchanged=True,scope=cfg['scope'])
    except Exception:
        result=dict(status='failed',started_utc=started,finished_utc=utc(),precision=a.precision,length=a.length,
                    rows=rows,gates=gates,scientific_optimizer_updates=0,error=traceback.format_exc())
        event('failed',error=result['error'])
    save(out/'result.json',result)
    if result['status']!='complete':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--precision',choices=['fp32','amp_bf16'],required=True)
    p.add_argument('--length',type=int,choices=[8192,16384,32768],required=True);p.add_argument('--kind',choices=['prefill','chunked'],required=True);main(p.parse_args())
