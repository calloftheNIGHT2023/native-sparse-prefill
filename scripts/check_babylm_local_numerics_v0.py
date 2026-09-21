"""Bounded tiny CPU/CUDA validation of fixed-local W. No scientific data/model.

Default is plan-only. --execute is required; an external process timeout is
required for paid CUDA use. Original GPU tolerances are fixed below.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = {'cpu_oracle_float64': {'atol': 1e-10, 'rtol': 1e-8},
              'cpu_cuda_fp32': {'atol': 1e-4, 'rtol': 1e-3},
              'gpu_full_support': {'atol': 1e-6, 'rtol': 1e-5},
              'checkpoint_replay': {'atol': 1e-6, 'rtol': 1e-5}}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_checks(report, device, max_wall):
    import torch
    from dataclasses import replace
    sys.path.insert(0, str(ROOT))
    from src.babylm_hybrid.config import HybridConfig
    from src.babylm_hybrid.model import build_model, parameter_counts
    from src.babylm_hybrid.attention import GlobalAttention, _normalization_dtype
    from src.babylm_hybrid.local_attention_v0 import (
        LocalGlobalAttention, LocalHybridLM, build_local_model, local_block_selection, CONTRACT_BUFFER, CONDITION)
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet
    counts = report['counts']
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    def bound():
        if time.monotonic()-started >= max_wall:
            raise RuntimeError('Bounded engineering deadline exceeded')
        if counts['model_forward_attempts'] > 12 or counts['backward_attempts'] > 10 or counts['engineering_optimizer_steps'] > 3:
            raise RuntimeError('Engineering call budget exceeded')
    def check(name, condition, **details):
        bound()
        report['checks'][name] = {'passed': bool(condition), **details}
        if not condition:
            raise AssertionError(name)
    def compare(a, b, tolerance):
        a, b = a.detach().cpu().double(), b.detach().cpu().double()
        if a.shape != b.shape:
            return {'passed': False, 'shape_mismatch': True}
        delta=(a-b).abs(); limit=tolerance['atol']+tolerance['rtol']*a.abs()
        return {'passed':bool(torch.isfinite(a).all() and torch.isfinite(b).all() and (delta<=limit).all()),
                'max_abs_error':float(delta.max()) if delta.numel() else 0., 'threshold':tolerance}
    def maps(a,b,tol):
        if set(a)!=set(b):return {'passed':False,'keys_differ':True}
        details={k:compare(a[k],b[k],tol) if a[k] is not None and b[k] is not None else {'passed':a[k] is b[k]} for k in a}
        return {'passed':all(x['passed'] for x in details.values()),'tensor_count':len(details),
                'failed_names':[k for k,v in details.items() if not v['passed']],
                'max_abs_error':max((v.get('max_abs_error',0.) for v in details.values()),default=0.),'threshold':tol}
    def gradients(m):return {n:p.grad for n,p in m.named_parameters()}
    def build(cfg,local=True):
        counts['model_constructions']+=1
        return (build_local_model if local else build_model)(cfg,'dense',20260917,20260918)
    def forward(m,ids,segments):
        bound();counts['model_forward_attempts']+=1
        counts['synthetic_input_tokens']+=ids.numel()
        v=m(ids,segment_ids=segments,aux_weight=1.0)
        counts['model_forward_calls']+=1
        counts[('cuda' if ids.is_cuda else 'cpu')+'_model_forward_calls']+=1
        counts['synthetic_loss_tokens']+=v.token_loss_count
        return v
    def backward(loss,component=False):
        key='component_' if component else ''
        counts[key+'backward_attempts']+=1
        loss.backward()
        counts[key+'backward_calls']+=1
    def step(opt):
        bound();counts['optimizer_step_attempts']+=1;opt.step();counts['engineering_optimizer_steps']+=1
    def analytic_mask(length,B,K):
        mask=torch.zeros(length,length,dtype=torch.bool)
        chosen=torch.zeros(length,length//B,dtype=torch.bool)
        for q in range(length):
            complete=[b for b in range(length//B) if (b+1)*B-1<=q]
            keep=complete[-K:]
            for b in keep:
                chosen[q,b]=True
                for key in range(b*B,(b+1)*B):mask[q,key]=True
            for key in range(len(complete)*B,q+1):mask[q,key]=True
        return chosen,mask
    cfg=HybridConfig()
    report['tiny_model_config']=cfg.to_dict()
    report['condition']=CONDITION
    report['hardware']={'device':device}
    report['runtime']={'torch':torch.__version__,'cuda_version':torch.version.cuda,'torch_num_threads':torch.get_num_threads(),
       'dtype':'float32' if device=='cuda' else 'component_float64_full_model_float32',
       'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
       'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
    if device=='cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('Exactly one CUDA device is required for this bounded gate')
        prop=torch.cuda.get_device_properties(0)
        report['hardware'].update(name=prop.name,uuid=str(prop.uuid),total_memory_bytes=prop.total_memory,
                                  capability=list(torch.cuda.get_device_capability(0)),visible_device_count=1)
    report['source_hashes']={str(ROOT/p):sha(ROOT/p) for p in [
       'scripts/check_babylm_local_numerics_v0.py','src/babylm_hybrid/local_attention_v0.py',
       'src/babylm_hybrid/attention.py','src/babylm_hybrid/model.py','src/babylm_hybrid/config.py']}
    gdnfile=inspect.getfile(Qwen3NextGatedDeltaNet)
    report['source_hashes'][gdnfile]=sha(gdnfile)

    geometry=[]
    for K in (2,64):
        component=GlobalAttention(replace(cfg,selected_complete_blocks=K));counts['component_constructions']+=1
        for length in (1,3,4,5,7,8,9,15,16,17,257,260,261,513):
            expected_chosen,expected_mask=analytic_mask(length,4,K)
            chosen,mask=local_block_selection(length,4,K)
            visible=torch.arange(length//4)[None,:]*4+3<=torch.arange(length)[:,None]
            scores=torch.arange(length*(length//4),dtype=torch.float32).reshape(length,length//4)
            _,sparsemask=component._sparse_selection(scores,visible);counts['component_support_calls']+=1
            expected_count=torch.tensor([4*min(K,(q+1)//4)+(q+1)%4 for q in range(length)])
            ok=torch.equal(mask,expected_mask) and torch.equal(chosen,expected_chosen) and torch.equal(mask.sum(-1),expected_count)
            ok=ok and torch.equal(sparsemask.sum(-1),expected_count) and not bool(torch.triu(mask,1).any()) and bool(mask.diag().all())
            geometry.append({'length':length,'K':K,'passed':ok,'logical_pairs':int(mask.sum())})
    check('independent_mask_per_query_budget_causal_tail',all(x['passed'] for x in geometry),cases=geometry,extra_model_calls=0)
    rejected=0
    for args in [(0,4,64),(5,0,64),(5,4,0),(True,4,64)]:
        try:local_block_selection(*args)
        except (TypeError,ValueError):rejected+=1
    check('invalid_support_arguments_rejected',rejected==4)

    rng=torch.get_rng_state().clone();dense=build(cfg,False);local=build(cfg)
    init_equal=all(torch.equal(p,dict(local.named_parameters())[n]) for n,p in dense.named_parameters())
    check('shared_backbone_initialization_rng_exact',init_equal and set(dict(dense.named_parameters()))==set(dict(local.named_parameters()))
          and torch.equal(rng,torch.get_rng_state()) and parameter_counts(local)==parameter_counts(dense)
          and parameter_counts(local)['indexer']==0 and all(p.requires_grad for p in local.parameters()),
          parameter_counts=parameter_counts(local),named_parameter_count=len(dict(local.named_parameters())))
    check('local_operator_active',local.mode=='local' and local.attention_condition==CONDITION
          and all(isinstance(l.mixer,LocalGlobalAttention) and l.mixer.indexer is None for l in local.layers if l.kind=='global'))
    rejected=0
    for action in [lambda:dense.load_state_dict(local.state_dict(),strict=True),
                   lambda:local.load_state_dict(dense.state_dict(),strict=True),
                   lambda:local.initialize_indexers(),lambda:build_local_model(cfg,'sparse',1,2)]:
        try:action()
        except (RuntimeError,ValueError):rejected+=1
    tampered=copy.deepcopy(local.state_dict());tampered[CONTRACT_BUFFER][1]+=1
    try:local.load_state_dict(tampered,strict=True)
    except RuntimeError:rejected+=1
    check('checkpoint_identity_and_budget_not_silently_dense',rejected==5)

    # Local component versus an independent enumerated token-mask oracle.
    dtype=torch.float64 if device=='cpu' else torch.float32
    oracle=GlobalAttention(cfg).to(dtype=dtype);actual=LocalGlobalAttention(cfg).to(device=device,dtype=dtype)
    counts['component_constructions']+=2
    actual.load_state_dict(oracle.state_dict(),strict=True)
    xin=torch.sin(torch.arange(21*cfg.hidden_size,dtype=dtype).reshape(21,cfg.hidden_size)/37).requires_grad_(True)
    xactual=xin.detach().to(device).clone().requires_grad_(True)
    labels=torch.tensor([[0]*13+[1]*5+[-1]*3])
    counts['component_forward_attempts']+=1
    out,aux,stats=actual(xactual[None],labels.to(device),mode='local')
    counts['component_forward_calls']+=1
    expected=xin*0;expected_pairs=0;expected_score=0
    for start,end in [(0,13),(13,18)]:
        x=xin[start:end];q,k,v=oracle._main_qkv(x);_,mask=analytic_mask(end-start,4,2)
        logits=torch.einsum('thd,shd->ths',q,k)*oracle.head_dim**-.5
        probs=logits.masked_fill(~mask[:,None,:],-torch.inf).softmax(-1,dtype=_normalization_dtype(logits)).to(v.dtype)
        val=oracle.out_proj(torch.einsum('ths,shd->thd',probs,v).reshape(end-start,-1)*oracle.gate_proj(x).sigmoid())
        expected=expected.index_copy(0,torch.arange(start,end),val)
        expected_pairs+=int(mask.sum());expected_score+=(end-start)**2*oracle.query_heads
    counts['component_forward_attempts']+=1;counts['component_forward_calls']+=1
    tol=THRESHOLDS['cpu_oracle_float64' if device=='cpu' else 'cpu_cuda_fp32']
    cmp=compare(expected,out[0],tol)
    check('component_independent_local_oracle_forward',cmp['passed'],comparison=cmp)
    backward(expected.square().sum(),True);backward(out.square().sum(),True)
    cmpg=maps(gradients(oracle),gradients(actual),tol);cmpx=compare(xin.grad,xactual.grad,tol)
    check('component_independent_local_oracle_backward',cmpg['passed'] and cmpx['passed'],parameters=cmpg,input_gradient=cmpx)
    check('component_doc_pad_aux_counts',stats['segments']==2 and stats['valid_tokens']==18 and stats['logical_kept_pairs']==expected_pairs
          and stats['allocated_main_score_elements']==expected_score and stats['indexer_score_elements']==0
          and stats['aux_nonempty_support_query_count']==0 and float(aux)==0. and not bool(out[0,18:].any())
          and stats['mode']=='local',stats=stats)

    # K64 includes all visible history here: full model must match D, including GDN.
    fullcfg=replace(cfg,selected_complete_blocks=64)
    d=build(fullcfg,False).to(device);w=build(fullcfg).to(device)
    ids=(torch.arange(42).reshape(2,21)*7%cfg.vocab_size).to(device)
    seg=torch.tensor([[0]*13+[1]*5+[-1]*3,[2]*3+[-1]+[2]*17],device=device)
    od=forward(d,ids,seg);ow=forward(w,ids,seg)
    cf=compare(od.logits,ow.logits,THRESHOLDS['gpu_full_support'])
    backward(od.loss);backward(ow.loss)
    cg=maps(gradients(d),gradients(w),THRESHOLDS['gpu_full_support'])
    check('full_support_full_model_matches_dense_F_B',cf['passed'] and cg['passed'] and od.token_loss_count==ow.token_loss_count
          and float(ow.aux_loss)==0.,logits=cf,gradients=cg)

    # Truly sparse W: future, other document and padding gradients must be zero.
    m=build(cfg).to(device);capture={}
    def embedding_hook(module,args,value):value.retain_grad();capture['embedding']=value
    hook=m.embedding.register_forward_hook(embedding_hook)
    out=forward(m,ids,seg);embedded=capture['embedding']
    backward(out.logits[0,11].square().sum());hook.remove()
    check('full_model_causal_and_crossdoc_gradient',not bool(embedded.grad[0,12:].any())
          and not bool(embedded.grad[1].any()) and bool(embedded.grad[0,:12].abs().sum()>0))
    changed=ids.clone();changed[0,12:]=(changed[0,12:]+31)%cfg.vocab_size;changed[1]=(changed[1]+13)%cfg.vocab_size
    with torch.no_grad():future=forward(m,changed,seg)
    check('full_model_causal_otherdoc_invariance',torch.equal(out.logits[0,:12],future.logits[0,:12]))
    changed=ids.clone();changed[0,:13]=(changed[0,:13]+11)%cfg.vocab_size
    with torch.no_grad():otherdoc=forward(m,changed,seg)
    check('full_model_GDN_document_reset',torch.equal(out.logits[0,13:18],otherdoc.logits[0,13:18])
          and torch.equal(out.logits[1],otherdoc.logits[1]) and not bool(out.logits[0,18:].any()))
    m.zero_grad(set_to_none=True)
    fitted=forward(m,ids,seg);backward(fitted.loss)
    gs=gradients(m)
    check('all_backbone_gradients_present_finite',all(p.requires_grad for p in m.parameters())
          and all(g is not None and bool(torch.isfinite(g).all()) for g in gs.values()),
          zero_gradient_names=[n for n,g in gs.items() if g is not None and not bool(g.any())])
    if device=='cuda':
        cpu=build(cfg);cpuout=forward(cpu,ids.cpu(),seg.cpu());backward(cpuout.loss)
        cg=maps(gradients(cpu),gs,THRESHOLDS['cpu_cuda_fp32']);cf=compare(cpuout.logits,fitted.logits,THRESHOLDS['cpu_cuda_fp32'])
        check('cpu_cuda_sparse_full_model',cg['passed'] and cf['passed'],logits=cf,gradients=cg)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.1,betas=(.9,.95));step(opt)
    # Deterministic same-step checkpoint/optimizer replay (all tiny synthetic).
    state=copy.deepcopy(m.state_dict());optim=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
    opt.zero_grad(set_to_none=True);out=forward(m,ids,seg);backward(out.loss);step(opt)
    restored=build(cfg).to(device);restored.load_state_dict(state,strict=True)
    ropt=torch.optim.AdamW(restored.parameters(),lr=3e-4,weight_decay=.1,betas=(.9,.95));ropt.load_state_dict(optim)
    torch.set_rng_state(rng);rout=forward(restored,ids,seg);backward(rout.loss);step(ropt)
    cp=maps(dict(m.named_parameters()),dict(restored.named_parameters()),THRESHOLDS['checkpoint_replay'])
    check('local_checkpoint_optimizer_replay',cp['passed'] and all(float(x['step'])==2 for x in ropt.state_dict()['state'].values()),parameters=cp)
    report['gate_boundary']='Tiny correctness only. CPU run is not CUDA authorization; CUDA gate still requires externally verified device isolation/runtime and hard timeout.'
    report['no_sparse_speed_claim']=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-wall-seconds',type=int,default=120)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'status':'plan_only','device':args.device,'maximum_model_forwards':12,'maximum_engineering_updates':3,'thresholds':THRESHOLDS}));return 0
    if not 1<=args.max_wall_seconds<=180 or args.output.exists():
        raise ValueError('Fresh output and 1..180 second bounded execution required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    counters=['model_constructions','component_constructions','component_support_calls','model_forward_attempts','model_forward_calls',
              'cpu_model_forward_calls','cuda_model_forward_calls','backward_attempts','backward_calls','optimizer_step_attempts',
              'engineering_optimizer_steps','component_forward_attempts','component_forward_calls','component_backward_attempts',
              'component_backward_calls','scientific_optimizer_steps','scientific_training_tokens','synthetic_input_tokens','synthetic_loss_tokens']
    report={'schema_version':1,'scope':'tiny_synthetic_local_W_numerics_engineering_only','started_utc':utc(),
            'device':args.device,'thresholds':THRESHOLDS,'counts':dict.fromkeys(counters,0),'checks':{}}
    start=time.monotonic()
    try:
        run_checks(report,args.device,args.max_wall_seconds)
        report.update(status='passed',passed=True)
    except BaseException as error:
        report.update(status='failed',passed=False,failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    report.update(completed_utc=utc(),elapsed_seconds=time.monotonic()-start)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'output':str(args.output),'checks':len(report['checks']),'counts':report['counts'],'elapsed_seconds':report['elapsed_seconds'],
                      'failure':report.get('failure',{}).get('message')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
