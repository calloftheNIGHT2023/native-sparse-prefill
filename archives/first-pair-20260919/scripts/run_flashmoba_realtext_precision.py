"""Frozen-model paired pooling intervention; no training and no new-algorithm claim."""
import argparse,hashlib,json,math,shutil,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from torch.nn.attention import sdpa_kernel,SDPBackend
from flash_moba import flash_moba_varlen_func
from flash_moba.triton_mean_pool import flash_topk_mean_pool,mean_pool_kernel
from flash_moba.flash_moba_interface import flash_moba_attn_varlen_func,decide_lg_block_m
import flash_moba_cuda

ROOT=Path(__file__).resolve().parents[1]
COMMIT='39d9ac043b271d046a2181a9991e99a26b67bca1'
LOCKED_POOL_CONFIG=None
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sparse(q,k,v,block,topk,pool,scale=None,details=False):
    n,h,d=q.shape;cu=torch.tensor([0,n],device=q.device,dtype=torch.int32)
    inp=k.float() if pool=='fp32' else k
    if LOCKED_POOL_CONFIG is None:
        means,cm,_=flash_topk_mean_pool(inp,cu,n,block)
    else:
        nb=math.ceil(n/block);cm=torch.tensor([0,nb],device=q.device,dtype=torch.int32)
        means=torch.zeros((nb,k.shape[1],d),device=k.device,dtype=inp.dtype)
        bn,warps,stages=LOCKED_POOL_CONFIG
        mean_pool_kernel.fn[(nb,1,k.shape[1])](inp,means,d,block,cu,cm,inp.stride(0),inp.stride(1),
                                            means.stride(0),means.stride(1),kBlockN=bn,num_warps=warps,num_stages=stages)
    means=means.to(k.dtype)
    offsets,counts,indices,values,ids=flash_moba_cuda.moba_fused_topk(q,means,cu,cu,cm,n,n,topk,block,True)
    indices=flash_moba_cuda.varlen_sort(offsets.flatten(),(offsets+counts).flatten(),indices)
    out=flash_moba_attn_varlen_func(q,k,v,cu,cu,n,n,offsets,counts,indices,
                                  decide_lg_block_m(topk,block,n,True),block,
                                  dropout_p=0.,softmax_scale=scale,causal=True)
    return (out,ids[...,:topk],means) if details else out


def main(a):
    global LOCKED_POOL_CONFIG
    out=a.output;out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();started=utc()
    rows=[];probes=[];gates=[];state=dict(mode='dense',topk=2,context=0,length=0,probe=False)
    data=a.data
    cfg_path=a.config or data/'config.json'
    cfg=json.loads(cfg_path.read_text())
    shutil.copy2(__file__,out/'source.py');shutil.copy2(cfg_path,out/'frozen-config.json')
    def event(kind,**kw):
        obj=dict(utc=utc(),elapsed_seconds=time.perf_counter()-tick,event=kind,**kw)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(obj)+'\n')
        print(json.dumps(obj),flush=True)
    try:
        for item in json.loads((data/'manifest.json').read_text())['files']:
            assert sha(data/item['path'])==item['sha256'],item['path']
        torch.set_num_threads(4);torch.manual_seed(cfg['selection_seed']);torch.backends.cuda.matmul.allow_tf32=False
        repo=ROOT/'third_party/flash-moba-official-20260915'
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==COMMIT
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
        save(out/'manifest.json',dict(started_utc=started,source_sha256=sha(Path(__file__)),
             config_sha256=sha(cfg_path),input_manifest_sha256=sha(data/'manifest.json'),
             torch=str(torch.__version__),transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
             cuda=torch.version.cuda,upstream_commit=COMMIT,extension_sha256=sha(Path(flash_moba_cuda.__file__)),
             scientific_optimizer_updates=0))
        event('start',contexts=cfg['contexts'],lengths=cfg['lengths'])

        with torch.inference_mode():
            for n,h,hk,d,b,t in [(257,2,2,64,128,2),(1024,8,8,64,128,4),(257,14,2,64,128,2)]:
                q=torch.randn(n,h,d,device='cuda',dtype=torch.bfloat16)
                k,v=[torch.randn(n,hk,d,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
                cu=torch.tensor([0,n],device='cuda',dtype=torch.int32)
                official=flash_moba_varlen_func(q,k,v,cu,cu,n,n,b,t,True)
                actual,_,_=sparse(q,k,v,b,t,'official',details=True)
                torch.testing.assert_close(actual,official,atol=0,rtol=0)
                fp,_,means=sparse(q,k,v,b,t,'fp32',details=True)
                ref=torch.stack([z.float().mean(0).to(k.dtype) for z in k.split(b)])
                torch.testing.assert_close(means,ref,atol=1e-5,rtol=.008)
                gates.append(dict(name='adapter_and_pool',length=n,official_output_bitwise_equal=True,
                                  fp32_pooled_rounding_max_abs=float((means.float()-ref.float()).abs().max())))
            q,k,v=[torch.randn(256,8,64,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
            allblocks=sparse(q,k,v,128,2,'official')
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):dense=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True)[0].transpose(0,1)
            torch.testing.assert_close(allblocks,dense,atol=.03,rtol=.03)
            gates.append(dict(name='all_blocks_flash',max_abs=float((allblocks-dense).abs().max())))
        save(out/'adapter-checks.json',gates)

        def attention(module,query,key,value,attention_mask,scaling=None,dropout=0.,**kwargs):
            assert query.shape[0]==1 and key.shape==value.shape and query.shape[-2]==key.shape[-2]
            assert dropout==0 and kwargs.get('head_mask') is None
            q,k,v=[z[0].transpose(0,1).to(torch.bfloat16).contiguous() for z in (query,key,value)]
            if state['probe']:
                for topk in cfg['topks']:
                    oa,ia,ma=sparse(q,k,v,cfg['block_size'],topk,'official',scaling,True)
                    ob,ib,mb=sparse(q,k,v,cfg['block_size'],topk,'fp32',scaling,True)
                    changed=(ia.sort(-1).values!=ib.sort(-1).values).any(-1)
                    diff=(oa.float()-ob.float())
                    probes.append(dict(context=state['context'],length=state['length'],layer=module.layer_idx,topk=topk,
                        changed_query_head_rows=int(changed.sum()),total_query_head_rows=changed.numel(),
                        output_relative_frobenius=float(diff.norm()/oa.float().norm().clamp_min(1e-12)),
                        max_output_abs=float(diff.abs().max()),pooled_max_abs=float((ma.float()-mb.float()).abs().max())))
                    if state['context']==0 and state['length']==max(cfg['lengths']) and module.layer_idx==0 and topk==cfg['topks'][0]:
                        torch.save(dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),official_ids=ia.cpu(),fp32_pool_ids=ib.cpu()),out/'fixed-real-qkv-example.pt')
            if state['mode']=='dense':
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    result=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True,dropout_p=0.,scale=scaling,enable_gqa=q.shape[1]!=k.shape[1])
                return result.transpose(1,2).to(query.dtype),None
            result=sparse(q,k,v,cfg['block_size'],state['topk'],state['mode'],scaling)
            return result[None].to(query.dtype),None

        model,loading=AutoModelForCausalLM.from_pretrained(data/'model',local_files_only=True,trust_remote_code=False,
            use_safetensors=True,attn_implementation='eager',dtype=getattr(torch,cfg.get('model_dtype','bfloat16')),output_loading_info=True)
        assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading.get('mismatched_keys'),loading
        model=model.cuda().eval();model.requires_grad_(False)
        ALL_ATTENTION_FUNCTIONS.register('flashmoba_pool_diagnostic',attention)
        tokens=np.load(data/'tokens.npy');arrays={}
        with torch.inference_mode():
            # Standard model implementation is a reference for the adapter, not only itself.
            small=torch.tensor(tokens[:1,:256],device='cuda')
            native=model(input_ids=small,use_cache=False).logits
            native_loss=float(F.cross_entropy(native[0].float(),torch.tensor(tokens[0,1:257],device='cuda')))
            model.config._attn_implementation='flashmoba_pool_diagnostic'
            # A full-model repeat gate before any intervention.
            small=torch.tensor(tokens[:1,:256],device='cuda')
            one=model(input_ids=small,use_cache=False).logits
            custom_loss=float(F.cross_entropy(one[0].float(),torch.tensor(tokens[0,1:257],device='cuda')))
            standard_rel=float((one.float()-native.float()).norm()/native.float().norm())
            assert standard_rel<.02 and abs(custom_loss-native_loss)<.05,(standard_rel,native_loss,custom_loss)
            gates.append(dict(name='standard_eager_model_vs_flash_adapter',relative_logit_norm=standard_rel,
                              standard_nll=native_loss,adapter_nll=custom_loss))
            del native
            two=model(input_ids=small,use_cache=False).logits
            torch.testing.assert_close(one,two,atol=0,rtol=0)
            gates.append(dict(name='fullmodel_dense_repeat',bitwise_equal=True,loaded_parameters=sum(p.numel() for p in model.parameters())))
            # Both blocks visible: this model-level control should agree within BF16 rounding.
            state.update(mode='official',topk=2)
            three=model(input_ids=small,use_cache=False).logits
            four=model(input_ids=small,use_cache=False).logits
            torch.testing.assert_close(three,four,atol=0,rtol=0)
            gates.append(dict(name='fullmodel_sparse_repeat',bitwise_equal=True))
            rel=float((three.float()-one.float()).norm()/one.float().norm())
            assert rel<.01,rel
            gates.append(dict(name='fullmodel_allblocks_flash',relative_logit_norm=rel,max_abs=float((three.float()-one.float()).abs().max())))
            save(out/'adapter-checks.json',gates)
            save(out/'pooling-execution.json',dict(autotune_cache_after_gates=[dict(key=str(k),config=str(v)) for k,v in mean_pool_kernel.cache.items()],
                                                  evaluation_pool_config=cfg.get('pool_config'),gates_used_original_autotune=True))
            LOCKED_POOL_CONFIG=cfg.get('pool_config')
            del one,two,three,four
            for length in cfg['lengths']:
                for context in range(cfg['contexts']):
                    ids=torch.tensor(tokens[context:context+1,:length],device='cuda')
                    targets=torch.tensor(tokens[context,1:length+1],device='cuda')
                    modes=[('dense',0)]+[(p,k) for k in cfg['topks'] for p in ['official','fp32']]
                    if cfg.get('reverse_sparse_order'):modes=modes[:1]+list(reversed(modes[1:]))
                    predictions={};nlls={}
                    for mode,topk in modes:
                        st=utc();t=time.perf_counter()
                        state.update(mode=mode,topk=topk,context=context,length=length,
                                     probe=mode=='dense' and context<cfg['probe_contexts'])
                        logits=model(input_ids=ids,use_cache=False).logits[0]
                        losses=F.cross_entropy(logits.float(),targets,reduction='none')
                        pred=logits.argmax(-1)
                        label=mode if mode=='dense' else f'{mode}_k{topk}'
                        predictions[label]=pred.cpu().numpy();nlls[label]=losses.cpu().numpy()
                        arrays[f'n{length}_c{context}_{label}_nll']=nlls[label]
                        arrays[f'n{length}_c{context}_{label}_argmax']=predictions[label]
                        row=dict(length=length,context=context,mode=label,started_utc=st,finished_utc=utc(),seconds=time.perf_counter()-t,
                                 mean_nll=float(losses.mean()),late_half_nll=float(losses[length//2:].mean()),
                                 token_accuracy=float((pred==targets).float().mean()))
                        rows.append(row);event('evaluation',**row)
                        del logits,losses,pred
                    for topk in cfg['topks']:
                        for r in rows[-4:]:
                            if r['mode']==f'fp32_k{topk}':
                                r['argmax_changed_from_official']=int((predictions[f'official_k{topk}']!=predictions[f'fp32_k{topk}']).sum())
                    save(out/'progress.json',dict(conditions=rows,probes=probes))
            np.savez_compressed(out/'per-token-results.npz',**arrays)
        rng=np.random.default_rng(cfg['bootstrap_seed']);summaries=[]
        for length in cfg['lengths']:
            for topk in cfg['topks']:
                group={mode:[r for r in rows if r['length']==length and r['mode']==mode] for mode in ['dense',f'official_k{topk}',f'fp32_k{topk}']}
                for metric in ['mean_nll','late_half_nll']:
                    a1=np.array([r[metric] for r in group[f'official_k{topk}']]);b1=np.array([r[metric] for r in group[f'fp32_k{topk}']]);delta=b1-a1
                    boot=delta[rng.integers(0,len(delta),size=(cfg['bootstrap_resamples'],len(delta)))].mean(1)
                    ci=np.quantile(boot,[.025,.975]).tolist()
                    summaries.append(dict(length=length,topk=topk,metric=metric,contexts=len(delta),
                        dense_mean=float(np.mean([r[metric] for r in group['dense']])),official_mean=float(a1.mean()),fp32_pool_mean=float(b1.mean()),
                        paired_mean_delta=float(delta.mean()),paired_context_bootstrap_95ci=ci,
                        passing_expansion_gate=length==max(cfg['lengths']) and metric=='mean_nll' and abs(float(delta.mean()))>.01 and ci[0]*ci[1]>0))
        passing=sum(r['passing_expansion_gate'] for r in summaries)
        result=dict(status='complete',started_utc=started,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,
                    conditions=rows,probes=probes,summaries=summaries,gate_checks=gates,
                    expand_precision_training=passing==len(cfg['topks']),scientific_optimizer_updates=0,
                    original_algorithm_claim=False,scope=cfg['scope'])
        save(out/'evaluation.json',result);event('complete',conditions=len(rows),probe_conditions=len(probes),expand_precision_training=result['expand_precision_training'])
    except Exception:
        save(out/'failure.json',dict(status='failed',started_utc=started,finished_utc=utc(),error=traceback.format_exc(),completed_conditions=len(rows)))
        event('failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/flashmoba-realtext-precision-v0')
    p.add_argument('--config',type=Path);main(p.parse_args())
