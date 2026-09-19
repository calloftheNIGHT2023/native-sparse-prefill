"""One predeclared paired LoRA trajectory with complete training/evaluation timing."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,json,math,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import run_flashmoba_realtext_precision as base
from run_pool_training_control import LoRALinear

ROOT=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    started=utc();tick=time.perf_counter();updates=0;trace=[];evaluations=[]
    cfg=json.loads((a.data/'config.json').read_text());cond=next(x for x in cfg['conditions'] if x['name']==a.condition)
    assert a.seed in cfg['initialization_seeds']
    save(out/'frozen-config.json',cfg);(out/'source.py').write_bytes(Path(__file__).read_bytes())
    def event(kind,**kw):
        row=dict(utc=utc(),elapsed_seconds=time.perf_counter()-tick,event=kind)
        row.update(kw)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        assert sha(a.data/'tokens.npz')==cfg['tokens_sha256']
        actual_sha=sha(Path(base.flash_moba_cuda.__file__))
        assert actual_sha==cfg['extension_sha256'][cond['extension']]
        assert json.loads((ROOT/'results/flashmoba-quality-cost-k16-gate-v0/result.json').read_text())['status']=='passed'
        if cond['extension']=='barrier':
            gate=json.loads((ROOT/'results/flashmoba-long-backward-barrier-v0/result.json').read_text())
            assert gate['deterministic_repeat_pass'] and gate['reference_tolerance_pass']
        torch.set_num_threads(4);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False
        base.LOCKED_POOL_CONFIG=tuple(cfg['pool_config'])
        data=np.load(a.data/'tokens.npz');assert data['train'].shape==(64,8193)
        model,info=AutoModelForCausalLM.from_pretrained(ROOT/cfg['model_path'],local_files_only=True,
            trust_remote_code=False,use_safetensors=True,attn_implementation='eager',dtype=torch.float32,output_loading_info=True)
        assert not info['missing_keys'] and not info['unexpected_keys'] and not info.get('mismatched_keys')
        model=model.cuda().eval();model.requires_grad_(False)
        small=torch.tensor(data['train'][:1,:256],device='cuda')
        with torch.no_grad():native=model(small,use_cache=False).logits
        torch.manual_seed(a.seed)
        for layer in model.model.layers:
            for name in ['q_proj','k_proj','v_proj','o_proj']:
                setattr(layer.self_attn,name,LoRALinear(getattr(layer.self_attn,name),cfg['lora_rank'],cfg['lora_alpha']))
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        assert sum(p.numel() for p in params.values())==1081344
        initial={n:p.detach().cpu().clone() for n,p in params.items()};torch.save(initial,out/'initial-adapters.pt')
        initial_digest=hashlib.sha256(b''.join(p.numpy().tobytes() for p in initial.values())).hexdigest()
        state=dict(pool='dense',topk=cond['topk'])
        def attention(module,query,key,value,attention_mask,scaling=None,dropout=0.,**kwargs):
            assert query.shape[0]==1 and key.shape==value.shape and dropout==0 and kwargs.get('head_mask') is None
            q,k,v=[z[0].transpose(0,1).to(torch.bfloat16).contiguous() for z in (query,key,value)]
            if state['pool']=='dense':
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    y=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],
                        is_causal=True,dropout_p=0.,scale=scaling,enable_gqa=True)
                return y.transpose(1,2).to(query.dtype),None
            y=base.sparse(q,k,v,cfg['block_size'],state['topk'],state['pool'],scaling)
            return y[None].to(query.dtype),None
        ALL_ATTENTION_FUNCTIONS.register('quality_cost_pilot',attention);model.config._attn_implementation='quality_cost_pilot'
        with torch.no_grad():custom=model(small,use_cache=False).logits
        targets=torch.tensor(data['train'][0,1:257],device='cuda')
        rel=float((custom-native).norm()/native.norm());n1=float(F.cross_entropy(native[0],targets));n2=float(F.cross_entropy(custom[0],targets))
        assert rel<.02 and abs(n1-n2)<.05,(rel,n1,n2)
        save(out/'adapter-gate.json',dict(relative_logits=rel,native_nll=n1,custom_nll=n2,passed=True))
        del native,custom,small,targets
        state.update(pool=cond['pool'])
        save(out/'manifest.json',dict(condition=cond,seed=a.seed,started_utc=started,torch=str(torch.__version__),
            gpu=torch.cuda.get_device_name(),extension_path=base.flash_moba_cuda.__file__,extension_sha256=actual_sha,
            source_sha256=sha(Path(__file__)),adapter_source_sha256=sha(Path(base.__file__)),
            lora_source_sha256=sha(ROOT/'scripts/run_pool_training_control.py'),tokens_sha256=cfg['tokens_sha256'],
            initial_adapters_tensor_sha256=initial_digest,trainable_parameters=1081344,deterministic_backward=False,
            protocol='FP32 frozen backbone with trainable FP32 LoRA, BF16 attention; all methods same data, initialization and optimizer.'))
        event('model_ready',condition=cond['name'],seed=a.seed,initial_adapters_sha256=initial_digest)
        def loss_for(window):
            ids=torch.tensor(window[:-1][None],device='cuda');target=torch.tensor(window[1:],device='cuda')
            return F.cross_entropy(model(ids,use_cache=False).logits[0],target)
        def evaluate(split,step):
            torch.cuda.synchronize();st=time.perf_counter()
            with torch.no_grad():values=[float(loss_for(w)) for w in data[split]]
            torch.cuda.synchronize();assert all(math.isfinite(v) for v in values)
            row=dict(split=split,step=step,nlls=values,mean_nll=float(np.mean(values)),seconds=time.perf_counter()-st,utc=utc())
            evaluations.append(row);save(out/'evaluations.json',evaluations);event('evaluation_complete',**row)
            return row
        evaluate('calibration',0)
        optimizer=torch.optim.AdamW(list(params.values()),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['eps'],
                                   weight_decay=cfg['weight_decay'],foreach=False,fused=False)
        torch.cuda.reset_peak_memory_stats()
        for i,w in enumerate(data['train']):
            torch.cuda.synchronize();st=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            loss=loss_for(w);assert bool(torch.isfinite(loss));loss.backward()
            grad=torch.cat([p.grad.detach().flatten() for p in params.values()])
            assert bool(torch.isfinite(grad).all());norm=float(grad.norm());optimizer.step();updates+=1
            torch.cuda.synchronize();seconds=time.perf_counter()-st
            row=dict(step=updates,train_nll=float(loss.detach()),gradient_norm=norm,seconds=seconds,utc=utc())
            trace.append(row);event('step_complete',**row);del loss,grad
            if updates in cfg['calibration_steps']:evaluate('calibration',updates)
            if updates in [16,32,64]:
                torch.save({n:p.detach().cpu().clone() for n,p in params.items()},out/f'adapters-step-{updates}.pt')
            save(out/'partial.json',dict(updates=updates,trace=trace,evaluations=evaluations))
        evaluate('report',updates)
        peak=torch.cuda.max_memory_allocated()
        final={n:p.detach().cpu().clone() for n,p in params.items()}
        delta=torch.cat([(final[n]-initial[n]).flatten() for n in params])
        result=dict(status='complete',condition=cond,seed=a.seed,started_utc=started,finished_utc=utc(),
            wall_seconds=time.perf_counter()-tick,scientific_optimizer_updates=updates,training_tokens=updates*cfg['length'],
            train_step_seconds_total=sum(x['seconds'] for x in trace),steady_step_seconds_median=float(np.median([x['seconds'] for x in trace[4:]])),
            trace=trace,evaluations=evaluations,peak_gpu_bytes=peak,final_delta_sha256=hashlib.sha256(delta.numpy().tobytes()).hexdigest(),
            final_delta_norm=float(delta.norm()),initial_adapters_tensor_sha256=initial_digest,
            tokens_sha256=cfg['tokens_sha256'],extension_sha256=actual_sha,
            interpretation='Exploratory paired LoRA adaptation on a pretrained dense base. Same-token pilot; not full-parameter native sparse pretraining, not an independent benchmark dataset or proven novel method.')
    except Exception:
        result=dict(status='failed',condition=cond,seed=a.seed,started_utc=started,finished_utc=utc(),
                    scientific_optimizer_updates=updates,trace=trace,evaluations=evaluations,error=traceback.format_exc())
        event('failed',error=result['error'])
    save(out/'result.json',result)
    save(out/'file-manifest.json',[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()])
    if result['status']!='complete':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--condition',required=True);p.add_argument('--seed',type=int,required=True);main(p.parse_args())
