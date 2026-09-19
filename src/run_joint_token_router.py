"""Fixed three-arm from-scratch online token-indexer diagnostic."""
import argparse,copy,hashlib,json,math,random,shutil,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from zoology_entry import ROOT,UPSTREAM,LanguageModel,multiquery_ar,set_determinism
from zoology.config import ModelConfig
from zoology_sparse_schedule import install
from frozen_routing import TokenIndexer
from joint_token_routing import JointTokenRouter
from run_frozen_router import now,save,sha,weights_sha,evaluate,value_swap

def make_model(base,initial,device):
    model=LanguageModel(ModelConfig(**base['model']));model.load_state_dict(initial);install(model,'native')
    model.to(device);model.backbone.embeddings.device=device;return model

def preflight(base,initial,device):
    set_determinism(123);a=make_model(base,initial,device);a.eval();b=copy.deepcopy(a)
    ix=torch.nn.ModuleList([TokenIndexer(128,16) for _ in range(2)]).to(device);route=JointTokenRouter(a,ix,'exact')
    x=torch.randint(0,256,(2,64),device=device);one=a(x);two=b(x)
    torch.testing.assert_close(one,two,rtol=2e-4,atol=2e-5)
    sum(route.losses).backward();assert all(p.grad is None for p in a.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ix.parameters());ix.zero_grad();route.reset()
    one.square().mean().backward();two.square().mean().backward()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=2e-3,atol=2e-6)
    assert all(p.grad is None for p in ix.parameters());route.restore()
    return dict(device=device,forward_max_error=float((one-two).abs().max()),main_gradient_equivalence=True,aux_gradient_isolation=True,technical_optimizer_updates=0)

def run_one(out,name,rank,route_mode,base,initial,data,device,cap):
    out.mkdir(exist_ok=False);timer=time.perf_counter();started=now();steps=0;seen=0;curves=[]
    set_determinism(123);model=make_model(base,initial,device);assert weights_sha(model.state_dict())==weights_sha(initial)
    set_determinism(2026091470);indexers=torch.nn.ModuleList([TokenIndexer(128,rank) for _ in range(2)]).to(device)
    torch.save(indexers.state_dict(),out/'initial-indexer.pt');route=JointTokenRouter(model,indexers,route_mode)
    main_opt=torch.optim.AdamW(model.parameters(),lr=base['learning_rate'],weight_decay=base['weight_decay'])
    ix_opt=torch.optim.AdamW(indexers.parameters(),lr=.001,weight_decay=0)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(main_opt,T_max=100,eta_min=0.)
    set_determinism(123);train={k:v.to(device) for k,v in data['train'].items()}
    save(out/'config.json',dict(name=name,rank=rank,route=route_mode,base=base,initial_backbone_hash=weights_sha(initial),indexer_parameters=sum(p.numel() for p in indexers.parameters()),epochs=40,batch=32,main_seed=123,indexer_seed=2026091470))
    def checkpoint():
        return dict(model=model.state_dict(),indexers=indexers.state_dict(),main_optimizer=main_opt.state_dict(),indexer_optimizer=ix_opt.state_dict(),scheduler=schedule.state_dict(),steps=steps,epoch=epoch,rank=rank,route=route_mode,config=base,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None,numpy_rng=np.random.get_state(),python_rng=random.getstate())
    with (out/'events.jsonl').open('w',encoding='utf-8',buffering=65536) as log:
        def event(kind,**kw):
            log.write(json.dumps(dict(utc=now(),event=kind,elapsed_seconds=time.perf_counter()-timer,**kw))+'\n')
            if kind!='optimizer_step':log.flush()
        event('start',method=name,initial_backbone_hash=weights_sha(initial));epoch=-1
        try:
            for epoch in range(40):
                cap()
                if time.perf_counter()-timer>720:raise TimeoutError('Per-run 12-minute cap')
                model.train();route.collect_aux=True;ce_sum=0.;kl_sum=0.;answers=0
                for first in range(0,len(train['inputs']),32):
                    main_opt.zero_grad();ix_opt.zero_grad();route.reset();x=train['inputs'][first:first+32];y=train['labels'][first:first+32]
                    logits=model(x);ce=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten());aux=sum(route.losses)
                    (ce+aux).backward();ce_scalar=float(ce);kl_scalar=float(aux)
                    main_grad=float(torch.stack([p.grad.square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt())
                    ix_grad=float(torch.stack([p.grad.square().sum() for p in indexers.parameters()]).sum().sqrt())
                    if not all(math.isfinite(v) for v in (ce_scalar,kl_scalar,main_grad,ix_grad)):raise FloatingPointError('Nonfinite joint training')
                    main_opt.step();ix_opt.step();steps+=1;seen+=len(x);count=int((y!=-100).sum());ce_sum+=ce_scalar*count;kl_sum+=kl_scalar*len(x);answers+=count
                    event('optimizer_step',step=steps,epoch=epoch,ce=ce_scalar,kl_sum_layers=kl_scalar,main_gradient_norm=main_grad,indexer_gradient_norm=ix_grad,main_lr=main_opt.param_groups[0]['lr'],indexer_lr=.001,input_tokens=seen*64,supervised_answers=seen*4,batch_examples=len(x))
                model.eval();route.collect_aux=False;route.reset();dev=evaluate(model,data['development'],device)
                row=dict(epoch=epoch+1,step=steps,train_nll=ce_sum/answers,mean_kl_sum_layers=kl_sum/10000,development_accuracy=dev['accuracy'],development_nll=dev['nll'])
                curves.append(row);event('epoch',**row);print(json.dumps(dict(method=name,**row)),flush=True);schedule.step()
                if (epoch+1)%5==0:torch.save(checkpoint(),out/'checkpoint.pt')
            result=dict(status='fit_complete',method=name,rank=rank,route=route_mode,started_utc=started,finished_utc=now(),backbone_optimizer_updates=steps,indexer_optimizer_updates=steps,input_tokens=seen*64,supervised_answers=seen*4,wall_seconds=time.perf_counter()-timer,initial_backbone_hash=weights_sha(initial),final_backbone_hash=weights_sha(model.state_dict()),curves=curves)
            save(out/'fit-result.json',result);event('fit_complete',step=steps)
        except Exception:
            event('failed',traceback=traceback.format_exc());torch.save(checkpoint(),out/'failure-checkpoint.pt');raise
        finally:route.restore()
    return result

def main(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);timer=time.perf_counter();started=now();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def interrupted(sig,frame):raise RuntimeError('Interrupted by signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def cap():
        if time.perf_counter()-timer>2700:raise TimeoutError('Overall 45-minute cap')
    snap=out/'source';snap.mkdir()
    for name in ['run_joint_token_router.py','joint_token_routing.py','frozen_routing.py','run_frozen_router.py','zoology_entry.py','zoology_sparse_schedule.py']:shutil.copy2(ROOT/'src'/name,snap/name)
    shutil.copytree(UPSTREAM,snap/'upstream',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'docs/joint-token-router-plan-2026-09-14.md',out/'preregistered-plan.md')
    source=torch.load(args.source/'shared-data-and-initialization.pt',map_location='cpu',weights_only=True)
    base=torch.load(args.source/'native/checkpoint.pt',map_location='cpu',weights_only=False)['config'];initial=source['initial_state']
    torch.manual_seed(2026091471);fresh_seg=multiquery_ar(vocab_size=256,num_examples=1024,input_seq_len=64,seed=2026091471,num_kv_pairs=4)
    fresh=dict(inputs=fresh_seg.inputs,labels=fresh_seg.labels);swapped,pairs=value_swap(fresh)
    old=[d['inputs'] for d in source['splits'].values()]
    frozen=torch.load(args.frozen/'data.pt',map_location='cpu',weights_only=True);old.extend([frozen['fresh']['inputs'],frozen['swapped']['inputs']])
    hashes={hashlib.sha256(row.numpy().tobytes()).hexdigest() for x in old for row in x};fresh_hashes=[hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in fresh['inputs']]
    assert len(set(fresh_hashes))==1024 and not set(fresh_hashes)&hashes
    for x,y in zip(fresh['inputs'],fresh['labels']):
        mapping={int(x[i]):int(x[i+1]) for i in range(0,8,2)};assert len(mapping)==4 and len(set(mapping.values()))==4 and int((y!=-100).sum())==4
        for p in torch.where(y!=-100)[0]:assert int(p)>=8 and mapping[int(x[p])]==int(y[p])
    data=dict(train=source['splits']['train'],development=source['splits']['validation'],fresh=fresh,swapped=swapped,swap_positions=pairs)
    torch.save(dict(initial_state=initial,**data),out/'data-and-initialization.pt')
    save(out/'data-audit.json',dict(fresh_rows=1024,fresh_seed=2026091471,fresh_disjoint_from_all_old_rows=True,fresh_labels_checked=True,initial_backbone_hash=weights_sha(initial)))
    save(out/'environment.json',dict(utc=started,python=sys.version,torch=str(torch.__version__),device=args.device,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,argv=sys.argv))
    save(out/'preflight.json',preflight(base,initial,args.device));completed=[]
    for name,rank,route in [('exact_r16_shadow',16,'exact'),('learned_r16',16,'learned'),('learned_r64',64,'learned')]:
        completed.append(run_one(out/name,name,rank,route,base,initial,data,args.device,cap));save(out/'progress.json',dict(status='fitting',utc=now(),completed=[r['method'] for r in completed]))
        if args.device=='cuda':torch.cuda.empty_cache()
    # Frozen test protocol: no further optimizer updates after any fresh evaluation.
    for r in completed:
        cap();folder=out/r['method'];ck=torch.load(folder/'checkpoint.pt',map_location=args.device,weights_only=False)
        model=make_model(base,initial,args.device);model.load_state_dict(ck['model']);model.eval()
        ix=torch.nn.ModuleList([TokenIndexer(128,r['rank']) for _ in range(2)]).to(args.device);ix.load_state_dict(ck['indexers']);route=JointTokenRouter(model,ix,r['route']);route.collect_aux=False
        r.update(fresh=evaluate(model,fresh,args.device),swapped=evaluate(model,swapped,args.device));route.restore()
        assert weights_sha(model.state_dict())==r['final_backbone_hash'];r['status']='complete';save(folder/'result.json',r)
    exact=completed[0];control_valid=all(exact[s]['accuracy']>=.99 for s in ('fresh','swapped'))
    for r in completed[1:]:r['learned_from_scratch_gate']=control_valid and all(r[s]['accuracy']>=.99 and r[s]['accuracy']>=exact[s]['accuracy']-.01 for s in ('fresh','swapped'));save(out/r['method']/'result.json',r)
    result=dict(status='complete',started_utc=started,finished_utc=now(),wall_seconds=time.perf_counter()-timer,control_valid=control_valid,backbone_optimizer_updates=sum(r['backbone_optimizer_updates'] for r in completed),indexer_optimizer_updates=sum(r['indexer_optimizer_updates'] for r in completed),runs=completed,scope='Single-seed 437760-parameter MQAR online KL token-indexer baseline with full dense detached teacher; not novel and no end-to-end sparse training/speed claim')
    save(out/'result.json',result);save(out/'SUCCESS.json',dict(utc=now(),control_valid=control_valid));save(out/'progress.json',dict(status='complete',utc=now()))
    save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(status='complete',control_valid=control_valid,runs=[dict(method=r['method'],fresh=r['fresh']['accuracy'],swapped=r['swapped']['accuracy'],gate=r.get('learned_from_scratch_gate')) for r in completed])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--frozen',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    try:main(a)
    except Exception:
        if a.output.exists():save(a.output/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()))
        raise
