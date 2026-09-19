"""Bounded author-MQAR training continuation / paired seed controls."""
import argparse, copy, json, math, random, shutil, signal, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from zoology_entry import ROOT, UPSTREAM, set_determinism
from router_author_control import author_configs, make_model, selected_logits, evaluate, generated
from zoology_sparse_schedule import install
from run_frozen_router import now, save, sha, weights_sha
from run_router_author_control import checkpoint_save

PLAN = ROOT/'docs/router-author-falsification-plan-2026-09-14.md'
OLD = ROOT/'results/router-author-control-v0'

def model_for(cfg, initial, device, method):
    m=make_model(cfg,initial,device)
    if method=='exact_native': install(m,'native')
    return m

def rng():
    return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],numpy_rng=np.random.get_state(),python_rng=random.getstate())

def restore_rng(c):
    torch.set_rng_state(c['torch_rng'].cpu())
    if c['cuda_rng']: torch.cuda.set_rng_state_all([v.cpu() for v in c['cuda_rng']])
    np.random.set_state(c['numpy_rng']);random.setstate(c['python_rng'])

def setup(cfg, method, initial, device, lr, checkpoint=None):
    m=model_for(cfg,initial,device,method)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.1)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=64)
    if checkpoint:
        m.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer']);sched.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint)
    return m,opt,sched

def assert_nested(a,b):
    if torch.is_tensor(a): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_nested(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_nested(x,y)
    else: assert a==b

def resume_test(device, destination):
    """Three disposable updates; independent uninterrupted-vs-restored trajectories."""
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=author_configs()[1];initial=torch.load(OLD/'initialization.pt',weights_only=True,map_location='cpu')
    m,o,s=setup(cfg,'exact_native',initial,device,.01);set_determinism(98761)
    d=generated(2026091700,2);x=d['inputs'].to(device);y=d['labels'].to(device)
    def step(m,o,s):
        m.train();o.zero_grad();loss=torch.nn.functional.cross_entropy(*selected_logits(m,x,y));loss.backward();o.step();s.step()
    step(m,o,s)
    saved=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict(),**rng()))
    destination.parent.mkdir(parents=True,exist_ok=True)
    path=destination.with_suffix('.pt');torch.save(saved,path)
    step(m,o,s);expected_rng=rng()
    c=torch.load(path,map_location='cpu',weights_only=False);b,p,t=setup(cfg,'exact_native',initial,device,.01,c);step(b,p,t)
    assert_nested(m.state_dict(),b.state_dict());assert_nested(o.state_dict(),p.state_dict());assert_nested(s.state_dict(),t.state_dict())
    assert torch.equal(expected_rng['torch_rng'],torch.get_rng_state())
    for a,z in zip(expected_rng['cuda_rng'],rng()['cuda_rng']): assert torch.equal(a,z)
    save(destination,dict(utc=now(),device=device,status='passed',technical_optimizer_updates=3,bitwise_parameters_and_optimizer_and_scheduler=True,dropout_rng_equal=True,scientific_updates=0,checkpoint_sha256=sha(path)))
    print(destination.read_text(encoding='utf-8'),flush=True)

def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);started=now();timer=time.perf_counter()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=author_configs()[1];initial=torch.load(OLD/'initialization.pt',map_location='cpu',weights_only=True)
    data=torch.load(OLD/'data.pt',map_location='cpu',weights_only=True)
    parent=args.parent.resolve() if args.parent else None;c=None;first_epoch=0;steps=0
    if parent:
        c=torch.load(parent,map_location='cpu',weights_only=False);assert c['epoch_boundary'];first_epoch=c['epoch']+1;steps=c['steps']
        assert steps==first_epoch*391 and first_epoch==19 and c['scheduler']['last_epoch']==19
        assert c['scheduler']['T_max']==64 and c['scheduler']['base_lrs']==[args.lr]
        assert args.method=='exact_native' and args.seed==123
    elif args.seed!=123:
        set_determinism(args.seed);seed_model=make_model(cfg,device='cpu');initial=copy.deepcopy(seed_model.state_dict());del seed_model
    source=out/'source';source.mkdir();shutil.copy2(PLAN,out/'plan.md')
    for p in (ROOT/'src').glob('*.py'):shutil.copy2(p,source/p.name)
    shutil.copytree(UPSTREAM,source/'upstream',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    torch.save(initial,out/'initialization.pt');save(out/'config.json',cfg)
    protected={str(OLD/'data.pt'):sha(OLD/'data.pt')}
    if parent:protected[str(parent)]=sha(parent)
    save(out/'provenance.json',dict(utc=started,method=args.method,seed=args.seed,lr=args.lr,parent=str(parent) if parent else None,protected=protected,plan_sha256=sha(out/'plan.md'),initialization_sha256=weights_sha(initial),starting_epoch=first_epoch,starting_steps=steps,final_epoch=64,wall_cap_seconds=args.wall_cap,technical_optimizer_updates=0))
    save(out/'source-manifest.json',[dict(path=p.relative_to(source).as_posix(),sha256=sha(p)) for p in sorted(source.rglob('*')) if p.is_file()])
    save(out/'environment.json',dict(utc=now(),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),python=sys.version,tf32=False))
    m,opt,sched=setup(cfg,args.method,initial,'cuda',args.lr,c)
    if not c:set_determinism(args.seed)
    train={k:v.cuda() for k,v in data['train'].items()};curves=[];epoch=first_epoch-1;prefix_steps=steps
    def ck(boundary):return dict(config=cfg,model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,steps=steps,epoch_boundary=boundary,method=args.method,seed=args.seed,**rng())
    try:
        with (out/'events.jsonl').open('w',encoding='utf-8',buffering=65536) as f:
            def event(kind,**kw):
                f.write(json.dumps(dict(utc=now(),event=kind,elapsed_seconds=time.perf_counter()-timer,**kw))+'\n')
                if kind!='optimizer_step':f.flush()
            event('start',starting_epoch=first_epoch,starting_steps=steps)
            for epoch in range(first_epoch,64):
                m.train();ce_sum=0.;correct=0;tick_epoch=time.perf_counter()
                for first in range(0,100000,256):
                    if time.perf_counter()-timer>args.wall_cap:raise TimeoutError('Registered per-fit wall cap')
                    tick=time.perf_counter();opt.zero_grad();x=train['inputs'][first:first+256];y=train['labels'][first:first+256]
                    logits,targets=selected_logits(m,x,y);loss=torch.nn.functional.cross_entropy(logits,targets);loss.backward()
                    norm=float(torch.stack([p.grad.square().sum() for p in m.parameters() if p.grad is not None]).sum().sqrt());value=float(loss.detach())
                    assert math.isfinite(value) and math.isfinite(norm);opt.step();torch.cuda.synchronize();steps+=1
                    ce_sum+=value*len(targets);correct+=int((logits.detach().argmax(-1)==targets).sum())
                    event('optimizer_step',global_step=steps,new_main_updates=steps-prefix_steps,epoch=epoch+1,ce=value,main_gradient_norm=norm,main_lr=opt.param_groups[0]['lr'],batch_examples=len(x),input_tokens=x.numel(),supervised_answers=len(targets),step_wall_seconds=time.perf_counter()-tick,indexer_updates=0)
                train_seconds=time.perf_counter()-tick_epoch;dev=evaluate(m,data['development'],'cuda');sched.step();checkpoint_save(out/'checkpoint.pt',ck(True))
                if epoch+1 in [19,24,32,48,64]:shutil.copy2(out/'checkpoint.pt',out/f'checkpoint_epoch{epoch+1:03d}.pt')
                row=dict(epoch=epoch+1,global_step=steps,new_main_updates=steps-prefix_steps,train_nll=ce_sum/1600000,train_accuracy=correct/1600000,development_accuracy=dev['accuracy'],development_nll=dev['nll'],train_seconds=train_seconds)
                curves.append(row);event('epoch',**row);save(out/'progress.json',dict(utc=now(),status='training',**row));print(json.dumps(row),flush=True)
        for path,digest in protected.items():assert sha(Path(path))==digest
        result=dict(status='complete',started_utc=started,finished_utc=now(),wall_seconds=time.perf_counter()-timer,method=args.method,seed=args.seed,lr=args.lr,starting_epoch=first_epoch,epochs=64,new_main_updates=steps-prefix_steps,prefix_main_updates=prefix_steps,total_main_updates=steps,input_tokens=(64-first_epoch)*25600000,supervised_answers=(64-first_epoch)*1600000,indexer_updates=0,technical_optimizer_updates=0,curves=curves,development_gate=curves[-1]['development_accuracy']>=.99,first_observed_development_crossing=next((r['epoch'] for r in curves if r['development_accuracy']>=.99),None),final_model_sha256=weights_sha(m.state_dict()),parent_files_unchanged=True)
        save(out/'fit-result.json',result);save(out/'SUCCESS.json',dict(utc=now()));save(out/'progress.json',dict(utc=now(),status='complete',final_development_accuracy=curves[-1]['development_accuracy']));print(json.dumps({k:v for k,v in result.items() if k!='curves'}),flush=True)
    except Exception:
        checkpoint_save(out/'failure-checkpoint.pt',ck(False));save(out/'FAILURE.json',dict(utc=now(),epoch=epoch,steps=steps,new_updates=steps-prefix_steps,traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--parent',type=Path);p.add_argument('--method',choices=['dense','exact_native'],default='exact_native');p.add_argument('--seed',type=int,default=123);p.add_argument('--lr',type=float,default=.01);p.add_argument('--wall-cap',type=float,default=1800);p.add_argument('--resume-test',choices=['cpu','cuda']);args=p.parse_args()
    def interrupt(sig,frame):raise RuntimeError('Signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    if args.resume_test:resume_test(args.resume_test,args.output.resolve())
    else:main(args)
