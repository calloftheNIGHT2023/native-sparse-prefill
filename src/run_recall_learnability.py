import argparse,json,math,shutil,time,traceback
from pathlib import Path
import torch
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
from far_recall import family,static_support,additive_mask,near_control,removed_control
from recall_supervision import add_training_queries,query_loss
from recall_binding import install_binding_control
from run_joint_pilot import utc,sha,save,tensor_digest
ROOT=Path(__file__).resolve().parents[1]

def dataset(cfg,split):
    xs=[]; ys=[]; metas=[]
    for i in range(cfg['families'][split]):
        x,y,meta=family(cfg,split,i); xs.append(x); ys.append(y); metas.append(meta)
    return torch.cat(xs),torch.cat(ys),metas

def prediction(model,x,mask=None):
    hidden=model.gpt_neox(x,attention_mask=mask,use_cache=False).last_hidden_state[:,-1]
    return model.embed_out(hidden)

@torch.no_grad()
def evaluate(model,data,cfg,mask=None):
    model.eval(); x,y,_=data; correct=0; open_correct=0; nll=0; predicted=[]; values=cfg['value_base']
    for start in range(0,len(y),cfg['batch_size']):
        logits=prediction(model,x[start:start+cfg['batch_size']],mask)
        labels=y[start:start+cfg['batch_size']]
        p=logits[:,values:values+cfg['num_values']].argmax(-1)
        predicted.extend(p.tolist()); correct+=int((p==labels).sum())
        open_correct+=int((logits.argmax(-1)==labels+values).sum())
        nll+=float(torch.nn.functional.cross_entropy(logits,labels+values,reduction='sum'))
    return dict(accuracy=correct/len(y),open_vocab_accuracy=open_correct/len(y),nll=nll/len(y),
        examples=len(y),families=len(data[2]),predictions=predicted)

def main(args):
    cfg=json.loads(args.config.read_text(encoding='utf-8')); out=args.output; out.mkdir(parents=True,exist_ok=False)
    started=utc(); timer=time.perf_counter(); torch.set_num_threads(4); torch.manual_seed(cfg['model_seed'])
    c=GPTNeoXConfig(vocab_size=cfg['vocab_size'],hidden_size=cfg['hidden_size'],num_hidden_layers=cfg['layers'],
        num_attention_heads=cfg['attention_heads'],intermediate_size=cfg['intermediate_size'],
        max_position_embeddings=cfg['sequence_length'],attention_dropout=0,hidden_dropout=0)
    c._attn_implementation='eager'; model=GPTNeoXForCausalLM(c).float(); initial=tensor_digest(model.state_dict())
    install_binding_control(model,cfg)
    streaming=cfg.get('training_mode')=='fresh_families'
    data={s:dataset(cfg,s) for s in cfg['families'] if s!='train' or not streaming}
    all_hashes=[m['family_sha256'] for s in data.values() for m in s[2]]
    assert len(set(all_hashes))==len(all_hashes)
    seen_hashes=set(all_hashes)
    save(out/'frozen-config.json',cfg); save(out/'model-config.json',c.to_dict())
    save(out/'data-families.json',{s:d[2] for s,d in data.items()})
    if streaming: save(out/'training-stream-rule.json',dict(split='train_stream',family_indices=[0,cfg['updates']*cfg['batch_size']],
        upper_endpoint_exclusive=True,one_new_family_per_example=True,variants='CPU generator model_seed+700, uniform answer class; recorded in events'))
    save(out/'environment.json',dict(device='cpu',torch=str(torch.__version__),started_utc=started))
    snapshot=out/'source-snapshot'; snapshot.mkdir()
    for name in ['run_recall_learnability.py','far_recall.py','sparse_reference.py','routing_rules.py','recall_supervision.py','recall_binding.py']:
        shutil.copy2(ROOT/'src'/name,snapshot/name)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
    g=torch.Generator().manual_seed(cfg['model_seed']+700); curves=[]; step=0; completed_updates=0; train_seconds=0
    def event(kind,**kw):
        with (out/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(utc=utc(),elapsed_seconds=time.perf_counter()-timer,event=kind,**kw))+'\n')
    def checkpoint(status):
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=completed_updates,
            torch_rng=torch.get_rng_state(),sampler_rng=g.get_state(),cfg=cfg,status=status),out/'checkpoint.pt')
    event('start',initial_hash=initial,scope=cfg['scope'])
    try:
        baseline=evaluate(model,data['validation'],cfg); curves.append(dict(step=0,**baseline))
        event('development_eval',step=0,accuracy=baseline['accuracy'])
        for step in range(1,cfg['updates']+1):
            if time.perf_counter()-timer>cfg['max_seconds']:
                checkpoint('paused_time_limit'); event('paused_time_limit',step=completed_updates)
                save(out/'status.json',dict(status='paused_time_limit',updates_completed=completed_updates)); return
            model.train()
            batch_hashes=[]; variants=[]
            if cfg.get('training_mode')=='fresh_families':
                fresh_x=[]; fresh_y=[]; query_targets=[]
                for b in range(cfg['batch_size']):
                    xx,yy,meta=family(cfg,'train_stream',(step-1)*cfg['batch_size']+b)
                    assert meta['family_sha256'] not in seen_hashes
                    seen_hashes.add(meta['family_sha256']); batch_hashes.append(meta['family_sha256'])
                    variant=int(torch.randint(cfg['num_values'],(1,),generator=g))
                    variants.append(variant)
                    if 'supervision' in cfg:
                        xx,qlabels=add_training_queries(xx,meta,cfg);query_targets.append(qlabels[variant])
                    fresh_x.append(xx[variant]); fresh_y.append(yy[variant])
                x=torch.stack(fresh_x); y=torch.stack(fresh_y)+cfg['value_base']
            else:
                idx=torch.randint(len(data['train'][1]),(cfg['batch_size'],),generator=g)
                x=data['train'][0][idx]; y=data['train'][1][idx]+cfg['value_base']
            ratio=min(1.,step/cfg['lr_warmup_updates'])
            if step>cfg['lr_warmup_updates']:
                phase=(step-cfg['lr_warmup_updates'])/(cfg['updates']-cfg['lr_warmup_updates'])
                ratio=.1+.9*(1+math.cos(math.pi*phase))/2
            opt.param_groups[0]['lr']=cfg['learning_rate']*ratio
            opt.zero_grad(set_to_none=True); before=time.perf_counter()
            loss=query_loss(model,x,torch.stack(query_targets),cfg) if 'supervision' in cfg else torch.nn.functional.cross_entropy(prediction(model,x),y)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True); opt.step(); completed_updates=step
            seconds=time.perf_counter()-before; train_seconds+=seconds
            event('optimizer_step',step=step,loss=loss.item(),grad_norm=float(norm),step_seconds=seconds,
                lr=opt.param_groups[0]['lr'],examples=step*cfg['batch_size'],input_tokens=step*cfg['batch_size']*cfg['sequence_length'],
                fresh_family_hashes=batch_hashes,fresh_family_variants=variants,
                supervised_answers=step*cfg['batch_size']*(cfg['records'] if cfg.get('supervision')=='all_queries' else 1))
            if step%cfg['evaluation_every']==0 or step==cfg['updates']:
                dev=evaluate(model,data['validation'],cfg); curves.append(dict(step=step,**dev))
                event('development_eval',step=step,accuracy=dev['accuracy']); checkpoint('running')
                print(json.dumps(dict(step=step,dev_accuracy=dev['accuracy'],loss=loss.item())),flush=True)
        _,_,local=static_support(cfg); mask=additive_mask(local)
        test=evaluate(model,data['test'],cfg); local_test=evaluate(model,data['test'],cfg,mask)
        assert local_test['accuracy']==1/cfg['num_values']
        for start in range(0,len(local_test['predictions']),cfg['num_values']):
            assert len(set(local_test['predictions'][start:start+cfg['num_values']]))==1
        near_x=[]; removed_x=[]
        for i,meta in enumerate(data['test'][2]):
            x=data['test'][0][i*cfg['num_values']:(i+1)*cfg['num_values']]
            near_x.append(near_control(x,meta,cfg)[0]); removed_x.append(removed_control(x,meta,cfg))
        near=evaluate(model,(torch.cat(near_x),data['test'][1],data['test'][2]),cfg,mask)
        removed=evaluate(model,(torch.cat(removed_x),data['test'][1],data['test'][2]),cfg)
        final=tensor_digest(model.state_dict()); assert initial!=final; checkpoint('complete')
        result=dict(status='complete',started_utc=started,finished_utc=utc(),updates=step,
            parameters=sum(p.numel() for p in model.parameters()),input_tokens=step*cfg['batch_size']*cfg['sequence_length'],
            supervised_answer_tokens=step*cfg['batch_size']*(cfg['records'] if cfg.get('supervision')=='all_queries' else 1),training_seconds=train_seconds,
            initial_hash=initial,final_hash=final,development=curves,dense_test=test,local_far_test=local_test,
            local_near_test=near,removed_evidence_dense_test=removed,
            learnability_gate_passed=curves[-1]['accuracy']>=cfg['learnability_gate_accuracy'] and test['accuracy']>=cfg['learnability_gate_accuracy'],
            scope=cfg['scope'],gpu_jobs=0,new_cloud_spend_usd=0)
        save(out/'result.json',result); event('complete',test_accuracy=test['accuracy'])
        save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in out.rglob('*') if p.is_file()])
        print(json.dumps({k:v for k,v in result.items() if k not in ['development','dense_test','local_far_test','local_near_test','removed_evidence_dense_test']}))
    except Exception:
        event('failed',traceback=traceback.format_exc()); checkpoint('failed'); raise
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
