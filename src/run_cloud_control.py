"""Bounded, resumable 2K CUDA dense control with full logs and fixed readouts.

This runner stops its process at the time limit, NOT RunPod billing.
"""
import argparse,contextlib,json,math,os,shutil,statistics,sys,time,traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from run_joint_pilot import build_model,utc,save,sha,tensor_digest
ROOT=Path(__file__).resolve().parents[1]

def sequence_order(count,updates,seed):
    order=[]; epoch=0
    while len(order)<updates:
        g=torch.Generator().manual_seed(seed+epoch)
        order.extend(torch.randperm(count,generator=g).tolist()); epoch+=1
    return order[:updates]

@torch.no_grad()
def evaluate(model,wrapper,tokens,control,cfg,device):
    model.eval(); previous=wrapper.cfg.get('selection_rule','learned')
    wrapper.cfg['selection_rule']='sink_recent'; rows=[]
    try:
        for row in tokens:
            ids=row.to(device); wrapper.reset('sparse' if control=='sink_recent' else control,False)
            logits=model(ids[:-1][None],use_cache=False).logits[0]
            nll=torch.nn.functional.cross_entropy(logits,ids[1:],reduction='none')
            rows.append(dict(all_nll=nll.mean().item(),late_nll=nll[cfg['late_query_start']:].mean().item()))
    finally: wrapper.cfg['selection_rule']=previous
    return dict(all_nll=statistics.mean(r['all_nll'] for r in rows),
        late_nll=statistics.mean(r['late_nll'] for r in rows),per_article=rows)

def main(args):
    cfg=json.loads(args.config.read_text()); out=args.output.resolve()
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA required, no CPU fallback')
    out.mkdir(parents=True,exist_ok=args.resume)
    if not args.resume and any(out.iterdir()): raise FileExistsError(out)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((args.data/'manifest.json').read_text())
    assert sha(args.data/'tokens.pt')==manifest['tokens_sha256']
    assert manifest['sequence_length']==cfg['sequence_length']
    upload=ROOT/'logs/upload-manifest.json'
    if upload.exists():
        for rec in json.loads(upload.read_text()):
            if rec['path'].startswith(cfg['pretrained_source']+'/'): assert sha(ROOT/rec['path'])==rec['sha256']
    tokens=torch.load(args.data/'tokens.pt',weights_only=True)
    order=sequence_order(len(tokens['train']),cfg['updates'],cfg['seed']+12000)
    config_hash=sha(args.config); data_hash=sha(args.data/'manifest.json')
    timer=time.perf_counter(); started=utc()
    def event(kind,**kw):
        record=dict(utc=utc(),session_elapsed_seconds=time.perf_counter()-timer,event=kind,**kw)
        with (out/'events.jsonl').open('a') as f: f.write(json.dumps(record)+'\n'); f.flush()
    model,old=build_model(cfg,cfg['initialization'],cfg['seed']); old.restore()
    # Move backbone first, then create paired indexers on the target device.
    from joint_attention import JointAttention
    model.to(args.device); torch.manual_seed(cfg['seed']+9000); wrapper=JointAttention(model,cfg)
    params=list(model.parameters()); opt=torch.optim.AdamW(params,lr=cfg['lm_learning_rate'],weight_decay=cfg['weight_decay'])
    step=0; train_seconds=0.; curves=[]; initial=tensor_digest(model.state_dict())
    if args.resume:
        ckpt=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert ckpt['config_hash']==config_hash and ckpt['data_hash']==data_hash
        model.load_state_dict(ckpt['model']); wrapper.indexers.load_state_dict(ckpt['indexers']); opt.load_state_dict(ckpt['optimizer'])
        torch.set_rng_state(ckpt['cpu_rng'])
        if args.device=='cuda': torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        step=ckpt['step']; train_seconds=ckpt['training_seconds']; curves=ckpt['curves']; initial=ckpt['initial_hash']
    else:
        save(out/'frozen-config.json',cfg); save(out/'data-order.json',order)
        save(out/'environment.json',dict(started_utc=started,device=args.device,torch=str(torch.__version__),
            cuda=torch.version.cuda,python=sys.version,config_sha256=config_hash,data_manifest_sha256=data_hash,
            tf32=False,precision='float32',cuda_deterministic_algorithms=False))
        snapshot=out/'source-snapshot'; snapshot.mkdir()
        for source in (ROOT/'src').glob('*.py'): shutil.copy2(source,snapshot/source.name)
    def checkpoint(status):
        blob=dict(model=model.state_dict(),indexers=wrapper.indexers.state_dict(),optimizer=opt.state_dict(),
            step=step,training_seconds=train_seconds,curves=curves,initial_hash=initial,
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else [],
            config_hash=config_hash,data_hash=data_hash,status=status)
        torch.save(blob,out/'checkpoint.partial.pt'); os.replace(out/'checkpoint.partial.pt',out/'checkpoint.pt')
        save(out/'status.json',dict(status=status,step=step,tokens=step*cfg['sequence_length'],updated_utc=utc(),training_seconds=train_seconds))
    event('resume' if args.resume else 'start',step=step,initial_hash=initial)
    try:
        if not curves:
            baseline={m:evaluate(model,wrapper,tokens['validation'],m,cfg,args.device) for m in ['dense','sink_recent','self_only']}
            curves.append(dict(step=0,controls=baseline)); event('development_eval',step=0,controls=baseline)
        while step<cfg['updates']:
            if time.perf_counter()-timer>=args.max_seconds:
                checkpoint('paused_time_limit'); event('paused_time_limit',step=step); return
            ids=tokens['train'][order[step]].to(args.device)
            model.train(); wrapper.reset('dense',False); opt.zero_grad(set_to_none=True)
            ratio=min(1.,(step+1)/cfg['lr_warmup_updates'])
            if step>=cfg['lr_warmup_updates']:
                phase=(step-cfg['lr_warmup_updates'])/max(1,cfg['updates']-cfg['lr_warmup_updates']-1)
                ratio=.1+.9*.5*(1+math.cos(math.pi*phase))
            opt.param_groups[0]['lr']=cfg['lm_learning_rate']*ratio
            if args.device=='cuda': torch.cuda.synchronize()
            before=time.perf_counter(); logits=model(ids[:-1][None],use_cache=False).logits[0]
            loss=torch.nn.functional.cross_entropy(logits,ids[1:])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True); opt.step()
            if args.device=='cuda': torch.cuda.synchronize()
            elapsed=time.perf_counter()-before; train_seconds+=elapsed; step+=1
            event('optimizer_step',step=step,row_index=order[step-1],lm_loss=loss.item(),gradient_norm=float(grad),
                lr=opt.param_groups[0]['lr'],step_seconds=elapsed,training_seconds=train_seconds,
                tokens=step*cfg['sequence_length'],peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
            if step%cfg['evaluation_every']==0 or step==cfg['updates']:
                controls={m:evaluate(model,wrapper,tokens['validation'],m,cfg,args.device) for m in ['dense','sink_recent','self_only']}
                curves.append(dict(step=step,controls=controls)); event('development_eval',step=step,controls=controls)
                checkpoint('running')
                print(json.dumps(dict(step=step,dev={k:v['late_nll'] for k,v in controls.items()})),flush=True)
        test={m:evaluate(model,wrapper,tokens['test'],m,cfg,args.device) for m in ['dense','sink_recent','self_only']}
        final=tensor_digest(model.state_dict()); assert final!=initial
        checkpoint('complete')
        gapdev=curves[-1]['controls']['sink_recent']['late_nll']-curves[-1]['controls']['dense']['late_nll']
        gaptest=test['sink_recent']['late_nll']-test['dense']['late_nll']
        result=dict(status='complete',finished_utc=utc(),updates=step,tokens=step*cfg['sequence_length'],
            training_seconds=train_seconds,initial_hash=initial,final_hash=final,development=curves,test=test,
            fixed_context_gate=dict(dev_gap_nats=gapdev,test_gap_nats=gaptest,threshold_nats=cfg['context_gate_nats'],
                passed=gapdev>=cfg['context_gate_nats'] and gaptest>=cfg['context_gate_nats']),
            scope=cfg['scope'],cloud_billing_stopped=False)
        save(out/'result.json',result); event('complete',step=step)
        save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'])
    except Exception:
        event('failed',step=step,traceback=traceback.format_exc())
        checkpoint('failed'); raise
    finally: wrapper.restore()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--max-seconds',type=float,default=3600); p.add_argument('--resume',action='store_true'); main(p.parse_args())
