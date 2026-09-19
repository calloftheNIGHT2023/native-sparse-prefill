"""Fixed-plan CPU joint LM/indexer pilot with raw logs and complete checkpoints."""
import argparse, hashlib, json, math, shutil, statistics, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoConfig, GPTNeoXForCausalLM
from joint_attention import JointAttention

ROOT=Path(__file__).resolve().parents[1]
def utc(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj): p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def tensor_digest(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def evaluate(model,wrapper,tokens,mode,cfg):
    model.eval(); wrapper.indexers.eval(); rows=[]
    for ids in tokens:
        wrapper.reset(mode,False)
        logits=model(ids[:-1][None],use_cache=False).logits[0]
        nll=torch.nn.functional.cross_entropy(logits,ids[1:],reduction='none')
        rows.append({'all_nll':nll.mean().item(),'late_nll':nll[cfg['late_query_start']:].mean().item()})
    return {'all_nll':statistics.mean(x['all_nll'] for x in rows),
            'late_nll':statistics.mean(x['late_nll'] for x in rows),'per_example':rows}

def build_model(cfg,initialization,seed):
    torch.manual_seed(seed)
    source=ROOT/cfg['pretrained_source']
    if initialization=='step1000':
        model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=False,
            use_safetensors=True,attn_implementation='eager').float()
    elif initialization=='random':
        c=AutoConfig.from_pretrained(source,local_files_only=True,trust_remote_code=False)
        c._attn_implementation='eager'; model=GPTNeoXForCausalLM(c).float()
    else: raise ValueError(initialization)
    torch.manual_seed(seed+9000)
    return model,JointAttention(model,cfg)

def run_one(cfg,initialization,seed,method,tokens,folder):
    folder.mkdir(); started=utc(); timer=time.perf_counter()
    def event(kind,**extra):
        with (folder/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'utc':utc(),'monotonic_seconds':time.perf_counter()-timer,'event':kind,**extra})+'\n')
    event('start',initialization=initialization,seed=seed,method=method)
    try:
        model,wrapper=build_model(cfg,initialization,seed)
        initial={'backbone':tensor_digest(model.state_dict()),'indexers':tensor_digest(wrapper.indexers.state_dict())}
        save(folder/'initialization-hashes.json',initial)
        lm_params=list(model.parameters()); index_params=list(wrapper.indexers.parameters())
        opt=torch.optim.AdamW([{'params':lm_params,'lr':cfg['lm_learning_rate']},
            {'params':index_params,'lr':cfg['indexer_learning_rate']}],weight_decay=cfg['weight_decay'])
        final_mode='dense' if method=='dense' else 'sparse'
        dev0=evaluate(model,wrapper,tokens['validation'],final_mode,cfg)
        curve=[{'step':0,'mode':final_mode,**dev0}]; event('development_eval',step=0,late_nll=dev0['late_nll'])
        lm_wall=0.; total_pairs=0
        for step in range(cfg['updates']):
            model.train(); wrapper.indexers.train()
            warmup=method=='sparse_warmup' and step<cfg['warmup_updates']
            mode='dense' if method=='dense' or warmup else 'sparse'
            wrapper.reset(mode,method!='dense')
            opt.zero_grad(set_to_none=True)
            ids=tokens['train'][step%len(tokens['train'])]
            ratio=min(1.,(step+1)/cfg['lr_warmup_updates'])
            if step>=cfg['lr_warmup_updates']:
                phase=(step-cfg['lr_warmup_updates'])/max(1,cfg['updates']-cfg['lr_warmup_updates']-1)
                ratio=.1+.9*.5*(1+math.cos(math.pi*phase))
            for group,base in zip(opt.param_groups,[cfg['lm_learning_rate'],cfg['indexer_learning_rate']]): group['lr']=base*ratio
            start_step=time.perf_counter()
            logits=model(ids[:-1][None],use_cache=False).logits[0]
            lm=torch.nn.functional.cross_entropy(logits,ids[1:])
            aux=torch.stack(wrapper.losses).mean() if wrapper.losses else lm.new_zeros(())
            loss=lm+cfg['aux_weight']*aux
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite joint loss')
            loss.backward()
            lm_grad=torch.nn.utils.clip_grad_norm_(lm_params,cfg['gradient_clip_norm'],error_if_nonfinite=True)
            idx_grad=torch.nn.utils.clip_grad_norm_(index_params,cfg['gradient_clip_norm'],error_if_nonfinite=True)
            opt.step(); lm_wall+=time.perf_counter()-start_step
            pairs=sum(x['retained_pairs'] for x in wrapper.stats); total_pairs+=pairs
            event('optimizer_step',step=step+1,mode=mode,lm_loss=lm.item(),aux_loss=aux.item(),
                lm_grad=float(lm_grad),indexer_grad=float(idx_grad),lr_ratio=ratio,
                tokens=(step+1)*cfg['sequence_length'],cpu_training_seconds=lm_wall,
                logical_main_attention_pairs=total_pairs,
                late_retained_fraction=statistics.mean(x['late_retained_fraction'] for x in wrapper.stats))
            if (step+1)%cfg['evaluation_every']==0 or step+1==cfg['updates']:
                dev=evaluate(model,wrapper,tokens['validation'],final_mode,cfg)
                curve.append({'step':step+1,'mode':final_mode,**dev})
                event('development_eval',step=step+1,late_nll=dev['late_nll'])
                print(json.dumps({'initialization':initialization,'method':method,'step':step+1,'dev_late_nll':dev['late_nll']}),flush=True)
        test=evaluate(model,wrapper,tokens['test'],final_mode,cfg)
        dense_test=test if method=='dense' else evaluate(model,wrapper,tokens['test'],'dense',cfg)
        final={'backbone':tensor_digest(model.state_dict()),'indexers':tensor_digest(wrapper.indexers.state_dict())}
        if final['backbone']==initial['backbone']: raise AssertionError('Backbone failed to update')
        torch.save({'model':model.state_dict(),'indexers':wrapper.indexers.state_dict(),'optimizer':opt.state_dict(),
            'step':cfg['updates'],'torch_rng':torch.get_rng_state()},folder/'final-checkpoint.pt')
        save(folder/'development-curve.json',curve)
        result={'initialization':initialization,'seed':seed,'method':method,'started_utc':started,'finished_utc':utc(),
            'elapsed_seconds':time.perf_counter()-timer,'cpu_training_seconds':lm_wall,'updates':cfg['updates'],
            'tokens':cfg['updates']*cfg['sequence_length'],'parameters':sum(p.numel() for p in lm_params),
            'indexer_parameters':sum(p.numel() for p in index_params),'initial_hashes':initial,'final_hashes':final,
            'final_dev':curve[-1],'test':test,'final_weights_dense_test':dense_test,
            'logical_main_attention_pairs':total_pairs,'status':'complete','scope':cfg['scope']}
        save(folder/'result.json',result); event('complete',test_late_nll=test['late_nll'])
        save(folder/'manifest.json',[{'path':p.name,'sha256':sha(p)} for p in folder.iterdir() if p.is_file()])
        wrapper.restore(); return result
    except Exception:
        event('failed',traceback=traceback.format_exc()); raise

def main(args):
    cfg=json.loads(args.config.read_text(encoding='utf-8')); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    started=utc(); torch.set_num_threads(cfg['threads']); torch.use_deterministic_algorithms(True)
    data=args.data.resolve(); manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    assert sha(data/'tokens.pt')==manifest['tokens_sha256']
    assert sha(args.config)==manifest['config_sha256']
    assets=ROOT/cfg['pretrained_source']; source_manifest=json.loads((assets.parents[1]/'download-manifest.json').read_text(encoding='utf-8'))
    for rec in source_manifest:
        if '/'+cfg['pretrained_revision']+'/' in rec['url']: assert sha(ROOT/rec['path'])==rec['sha256']
    tokens=torch.load(data/'tokens.pt',weights_only=True)
    save(out/'frozen-config.json',cfg); save(out/'environment.json',{'started_utc':started,'torch':torch.__version__,
        'python':sys.version,'device':'cpu','data_manifest_sha256':sha(data/'manifest.json')})
    snapshot=out/'source-snapshot'; snapshot.mkdir()
    for name in ['run_joint_pilot.py','joint_attention.py','indexer_calibration.py','sparse_reference.py','trace_math.py','gathered_core.py','routing_rules.py']:
        shutil.copy2(ROOT/'src'/name,snapshot/name)
    results=[]
    for initialization in cfg['initializations']:
        for seed in cfg['seeds']:
            for method in cfg['methods']:
                results.append(run_one(cfg,initialization,seed,method,tokens,out/f'{initialization}__seed{seed}__{method}'))
    gates=[]
    for initialization in cfg['initializations']:
        rows={r['method']:r for r in results if r['initialization']==initialization}
        a,b=rows['sparse_step0'],rows['sparse_warmup']
        dev_gap=a['final_dev']['late_nll']-b['final_dev']['late_nll']; test_gap=a['test']['late_nll']-b['test']['late_nll']
        gates.append({'initialization':initialization,'dev_gap_nats':dev_gap,'test_gap_nats':test_gap,
            'local_mechanism_gate':dev_gap>=.02 and test_gap>=.02,'cloud_gate':False})
    save(out/'summary.json',{'started_utc':started,'finished_utc':utc(),'runs_completed':len(results),
        'gates':gates,'results':results,'scope':cfg['scope']})
    save(out/'manifest.json',[{'path':str(p.relative_to(out)).replace('\\','/'),'sha256':sha(p)} for p in out.rglob('*') if p.is_file()])

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); main(p.parse_args())
