"""Bounded 8K LoRA integration gate for an independently validated backward synchronization fix."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,functools,hashlib,inspect,json,math,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import run_flashmoba_realtext_precision as base
from verify_flashmoba_official_v1 import block_masks,reference
from triton.language import standard

ROOT=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def compare(a,b):
    a=a.double().flatten();b=b.double().flatten();delta=b-a
    return dict(relative_l2=float(delta.norm()/a.norm().clamp_min(1e-30)),cosine=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30)),max_abs=float(delta.abs().max()),changed_elements=int((a!=b).sum()),elements=a.numel())

class LoRALinear(torch.nn.Module):
    def __init__(self,layer,rank,alpha):
        super().__init__();self.base=layer;self.scale=alpha/rank
        self.A=torch.nn.Parameter(torch.empty(rank,layer.in_features,device=layer.weight.device,dtype=layer.weight.dtype))
        self.B=torch.nn.Parameter(torch.zeros(layer.out_features,rank,device=layer.weight.device,dtype=layer.weight.dtype))
        torch.nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
    def forward(self,x):return self.base(x)+(x@self.A.T@self.B.T)*self.scale

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();started=utc();rows=[];updates=0
    cfg=json.loads((a.data/'config.json').read_text());save(out/'frozen-config.json',cfg)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    def event(kind,**kw):
        row=dict(utc=utc(),elapsed_seconds=time.perf_counter()-tick,event=kind);row.update(kw)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        gate=json.loads((ROOT/'results/flashmoba-long-backward-barrier-v0/result.json').read_text())
        standard_gate=json.loads((ROOT/'results/flashmoba-newpod-standard-barrier-v0/verification.json').read_text())
        assert gate['status']=='complete' and gate['actual_k4_selected_masks_match']
        assert gate['deterministic_repeat_pass'] and gate['reference_tolerance_pass']
        assert standard_gate['status']=='passed' and len(standard_gate['checks'])==12
        assert sha(Path(base.flash_moba_cuda.__file__))==gate['extension_sha256']
        torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
        repo=ROOT/'third_party/flash-moba-official-20260915'
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==base.COMMIT
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
        (out/'triton-sum-source.txt').write_text(inspect.getsource(standard.sum.fn)+'\n'+inspect.getsource(getattr(standard._pick_sum_dtype,'fn',standard._pick_sum_dtype)))
        save(out/'manifest.json',dict(started_utc=started,source_sha256=sha(Path(__file__)),adapter_source_sha256=sha(Path(base.__file__)),
            config_sha256=sha(a.data/'config.json'),torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),upstream_commit=base.COMMIT,
            extension_sha256=sha(Path(base.flash_moba_cuda.__file__)),model_manifest_sha256=sha(ROOT/'data/flashmoba-qwen-precision-v0/manifest.json')))
        base.flash_moba_attn_varlen_func=functools.partial(base.flash_moba_attn_varlen_func,deterministic=cfg['deterministic_backward'])
        base.LOCKED_POOL_CONFIG=(32,4,3)
        # Independent FP32 masked reference for the GQA shape used by Qwen.
        q=torch.randn(257,14,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn(257,2,64,device='cuda',dtype=torch.bfloat16,requires_grad=True);v=torch.randn_like(k,requires_grad=True)
        actual,_,means=base.sparse(q,k,v,128,2,'official',details=True)
        masks,_=block_masks(q,k,[257],128,2,provided_means=means)
        qr,kr,vr=[x.detach().float().requires_grad_() for x in (q,k,v)]
        expected=reference(qr,kr,vr,[257],masks,128);go=torch.randn_like(actual)
        ga=torch.autograd.grad(actual,(q,k,v),go);gr=torch.autograd.grad(expected,(qr,kr,vr),go.float())
        torch.testing.assert_close(actual.float(),expected,atol=.03,rtol=.03)
        for x,y in zip(ga,gr):torch.testing.assert_close(x.float(),y,atol=.03,rtol=.03)
        second=base.sparse(q,k,v,128,2,'official');gb=torch.autograd.grad(second,(q,k,v),go)
        gates=[dict(name='gqa14_2_backward_reference',max_grad_abs=max(float((x.float()-y).abs().max()) for x,y in zip(ga,gr)),
                    repeat_gradients=[compare(x.float(),y.float()) for x,y in zip(ga,gb)],deterministic_requested=cfg['deterministic_backward'])]
        save(out/'gates.json',gates);event('backward_gate_passed',gates=gates)
        del q,k,v,actual,expected,qr,kr,vr,ga,gb,gr,go,second
        tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg['model_path'],local_files_only=True,trust_remote_code=False)
        arrays={};selected={};width=cfg['length']+1
        for split,count in [('train',cfg['steps']),('validation',cfg['validation_windows'])]:
            source=next(x for x in cfg['sources'] if x['split']==split)
            assert sha(a.data/f'{split}.txt')==source['text_sha256']
            ids=tokenizer((a.data/f'{split}.txt').read_text(encoding='utf-8'),add_special_tokens=False)['input_ids']
            windows=np.asarray(ids[:len(ids)//width*width],dtype=np.int64).reshape(-1,width)
            index=np.random.default_rng(cfg['seed']+(split=='validation')).permutation(len(windows))[:count]
            assert len(index)==count
            arrays[split]=windows[index];selected[split]=dict(indices=index.tolist(),corpus_token_count=len(ids))
        np.savez(out/'input-tokens.npz',**arrays);save(out/'input-manifest.json',dict(selected=selected,sha256=sha(out/'input-tokens.npz'),no_added_special_tokens=True))
        model,info=AutoModelForCausalLM.from_pretrained(ROOT/cfg['model_path'],local_files_only=True,trust_remote_code=False,
            use_safetensors=True,attn_implementation='eager',dtype=torch.float32,output_loading_info=True)
        assert not info['missing_keys'] and not info['unexpected_keys'] and not info.get('mismatched_keys')
        model=model.cuda().eval();model.requires_grad_(False)
        small=torch.tensor(arrays['train'][:1,:256],device='cuda')
        with torch.no_grad():native=model(small,use_cache=False).logits
        torch.manual_seed(cfg['seed'])
        for layer in model.model.layers:
            for name in ['q_proj','k_proj','v_proj','o_proj']:
                setattr(layer.self_attn,name,LoRALinear(getattr(layer.self_attn,name),cfg['lora_rank'],cfg['lora_alpha']))
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        initial={n:p.detach().cpu().clone() for n,p in params.items()};torch.save(initial,out/'initial-adapters.pt')
        state=dict(mode='dense',capture=False,routes=[])
        def attention(module,query,key,value,attention_mask,scaling=None,dropout=0.,**kwargs):
            assert query.shape[0]==1 and dropout==0 and kwargs.get('head_mask') is None
            q,k,v=[z[0].transpose(0,1).to(torch.bfloat16).contiguous() for z in (query,key,value)]
            if state['mode']=='dense':
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    output=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True,dropout_p=0.,scale=scaling,enable_gqa=True)
                return output.transpose(1,2).to(query.dtype),None
            if state['capture']:
                output,ids,_=base.sparse(q,k,v,cfg['block_size'],cfg['topk'],state['mode'],scaling,True)
                state['routes'].append(ids.detach().sort(-1).values.cpu())
            else:output=base.sparse(q,k,v,cfg['block_size'],cfg['topk'],state['mode'],scaling)
            return output[None].to(query.dtype),None
        ALL_ATTENTION_FUNCTIONS.register('pool_training_control',attention);model.config._attn_implementation='pool_training_control'
        with torch.no_grad():custom=model(small,use_cache=False).logits
        native_loss=F.cross_entropy(native[0],torch.tensor(arrays['train'][0,1:257],device='cuda'))
        custom_loss=F.cross_entropy(custom[0],torch.tensor(arrays['train'][0,1:257],device='cuda'))
        rel=float((custom-native).norm()/native.norm())
        assert rel<.02 and abs(float(custom_loss-native_loss))<.05,(rel,float(native_loss),float(custom_loss))
        gates.append(dict(name='standard_eager_vs_zero_lora_flash',relative_logits=rel,native_nll=float(native_loss),custom_nll=float(custom_loss)))
        save(out/'gates.json',gates);del native,custom,native_loss,custom_loss
        event('model_ready',trainable_parameters=sum(p.numel() for p in params.values()),total_parameters=sum(p.numel() for p in model.parameters()))
        def loss_for(window):
            ids=torch.tensor(window[:-1][None],device='cuda');targets=torch.tensor(window[1:],device='cuda')
            logits=model(ids,use_cache=False).logits[0]
            return F.cross_entropy(logits,targets)
        def validation():
            with torch.no_grad():return [float(loss_for(x)) for x in arrays['validation']]
        firstgrads={};deltas={};routes={};validation_rows=[]
        # All conditions start at exactly the same adapter bytes; base weights never change.
        for cond in cfg['conditions']:
            ct=time.perf_counter();cstart=utc();name=cond['name'];folder=out/name;folder.mkdir()
            with torch.no_grad():
                for n,p in params.items():p.copy_(initial[n])
            torch.manual_seed(cfg['seed']);optimizer=torch.optim.AdamW(list(params.values()),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'],foreach=False,fused=False)
            state.update(mode=cond['mode'],capture=False,routes=[]);base.LOCKED_POOL_CONFIG=(cond['bn'],4,3)
            before=validation();trace=[];torch.cuda.reset_peak_memory_stats()
            for step,window in enumerate(arrays['train']):
                st=time.perf_counter();optimizer.zero_grad(set_to_none=True);state.update(capture=step==0,routes=[])
                loss=loss_for(window);assert bool(torch.isfinite(loss));loss.backward()
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params.values())
                grad=torch.cat([p.grad.detach().flatten() for p in params.values()]);gnorm=float(grad.norm())
                if step==0:
                    firstgrads[name]=grad.cpu();torch.save({n:p.grad.detach().cpu() for n,p in params.items()},folder/'first-gradients.pt')
                    routes[name]=torch.stack(state['routes']) if state['routes'] else torch.empty(0,dtype=torch.int32)
                    torch.save(routes[name],folder/'first-routes.pt')
                optimizer.step();updates+=1;torch.cuda.synchronize()
                row=dict(step=step,train_nll=float(loss.detach()),gradient_norm=gnorm,seconds=time.perf_counter()-st,utc=utc())
                trace.append(row);event('step_complete',condition=name,**row)
                del loss,grad
            state.update(capture=False,routes=[]);after=validation()
            final={n:p.detach().cpu().clone() for n,p in params.items()};torch.save(final,folder/'final-adapters.pt')
            deltas[name]=torch.cat([(final[n]-initial[n]).flatten() for n in params])
            # Evaluate each trained sparse checkpoint with all three execution configurations.
            cross={}
            for bn in [32,64,128]:
                base.LOCKED_POOL_CONFIG=(bn,4,3);cross[str(bn)]=validation()
            row=dict(**cond,started_utc=cstart,finished_utc=utc(),wall_seconds=time.perf_counter()-ct,updates=len(trace),trace=trace,
                validation_before=before,validation_after=after,cross_configuration_validation=cross,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),first_gradient_sha256=tensor_sha(firstgrads[name]),final_delta_sha256=tensor_sha(deltas[name]))
            rows.append(row);save(folder/'trajectory.json',row);save(out/'partial.json',rows)
            event('condition_complete',condition=name,validation_before=float(np.mean(before)),validation_after=float(np.mean(after)),seconds=row['wall_seconds'])
            del optimizer,final
        pairs=[]
        for left,right,kind in cfg['comparison_pairs']:
            rr=dict(left=left,right=right,kind=kind,first_gradient=compare(firstgrads[left],firstgrads[right]),final_parameter_delta=compare(deltas[left],deltas[right]))
            if routes[left].numel():
                changed=(routes[left]!=routes[right]).any(-1);rr.update(first_forward_changed_route_rows=int(changed.sum()),first_forward_route_rows=changed.numel())
            pairs.append(rr)
        result=dict(status='complete',started_utc=started,finished_utc=utc(),seconds=time.perf_counter()-tick,scientific_optimizer_updates=updates,
            trajectories=rows,pairs=pairs,trainable_parameters=sum(p.numel() for p in params.values()),
            interpretation='Diagnostic only. Same-seed fixed-data execution interventions, not independent training replications or a quality benchmark. Literature already overlaps generic sparse-routing numerical mismatch.')
    except Exception:
        result=dict(status='failed',started_utc=started,finished_utc=utc(),seconds=time.perf_counter()-tick,scientific_optimizer_updates=updates,trajectories=rows,error=traceback.format_exc())
        event('failed',error=result['error'])
    save(out/'result.json',result)
    save(out/'file-manifest.json',[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()])
    if result['status']!='complete':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
