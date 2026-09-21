"""Fixed 19-epoch known-recipe comparison after a separately audited dense positive control."""
import argparse,json,math,os,random,shutil,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from zoology_entry import ROOT,UPSTREAM,set_determinism
from router_author_control import author_configs,generated,validate,row_hashes,interventions,selected_logits,evaluate,make_model
from router_author_sparse import build,initial_indexers,METHODS
from run_frozen_router import now,save,sha,weights_sha
from run_router_author_control import checkpoint_save

def run_one(out,method,cfg,initial,initial_ix,train_data,development,batch_timer):
    out.mkdir(exist_ok=False);timer=time.perf_counter();started=now();m,ix,route=build(cfg,initial,initial_ix,method,'cuda');assert weights_sha(m.state_dict())==weights_sha(initial)
    if len(ix):assert weights_sha(ix.state_dict())==weights_sha(initial_ix)
    opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.1);iopt=torch.optim.AdamW(ix.parameters(),lr=.001,weight_decay=0) if len(ix) else None;sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=64);set_determinism(123)
    train={k:v.cuda() for k,v in train_data.items()};epoch=-1;steps=0;curves=[];cost={k:0 for k in ['main_score_entries','index_score_entries','teacher_distribution_entries']};seconds_total=0.
    save(out/'config.json',dict(method=method,config=cfg,initial_backbone_sha=weights_sha(initial),initial_indexer_sha=weights_sha(initial_ix) if len(ix) else None,main_parameters=sum(p.numel() for p in m.parameters()),indexer_parameters=sum(p.numel() for p in ix.parameters()),target_epochs=19,indexer_learning_rate=.001,indexer_weight_decay=0,indexer_seed=2026091620))
    def ck(boundary):return dict(method=method,config=cfg,model=m.state_dict(),indexers=ix.state_dict(),main_optimizer=opt.state_dict(),indexer_optimizer=iopt.state_dict() if iopt else None,scheduler=sched.state_dict(),epoch=epoch,steps=steps,epoch_boundary=boundary,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    with (out/'events.jsonl').open('w',encoding='utf-8',buffering=65536) as f:
        def event(kind,**kw):
            f.write(json.dumps(dict(utc=now(),event=kind,elapsed_seconds=time.perf_counter()-timer,**kw))+'\n')
            if kind!='optimizer_step':f.flush()
        event('start',method=method)
        try:
            for epoch in range(19):
                route.epoch=epoch;route.collect_aux=True;m.train();ce_sum=0.;aux_sum=0.;correct=0;epoch_secs=0.
                for first in range(0,100000,256):
                    if time.perf_counter()-timer>1500 or time.perf_counter()-batch_timer>3900:raise TimeoutError('Author sparse preregistered wall cap')
                    tick=time.perf_counter();opt.zero_grad()
                    if iopt:iopt.zero_grad()
                    route.reset();x=train['inputs'][first:first+256];y=train['labels'][first:first+256];logits,targets=selected_logits(m,x,y);ce=torch.nn.functional.cross_entropy(logits,targets);aux=sum(route.losses);(ce+aux).backward()
                    mg=float(torch.stack([p.grad.square().sum() for p in m.parameters() if p.grad is not None]).sum().sqrt());ig=float(torch.stack([p.grad.square().sum() for p in ix.parameters() if p.grad is not None]).sum().sqrt()) if iopt else 0.;cv=float(ce.detach());av=float(aux.detach()) if isinstance(aux,torch.Tensor) else float(aux);assert all(math.isfinite(z) for z in [mg,ig,cv,av]);opt.step()
                    if iopt:iopt.step()
                    torch.cuda.synchronize();seconds=time.perf_counter()-tick;steps+=1;seconds_total+=seconds;epoch_secs+=seconds;ce_sum+=cv*len(targets);aux_sum+=av*len(x);correct+=int((logits.detach().argmax(-1)==targets).sum());bc={k:sum(c[k] for c in route.cost) for k in cost}
                    for k in cost:cost[k]+=bc[k]
                    event('optimizer_step',global_step=steps,epoch=epoch+1,batch_examples=len(x),input_tokens=x.numel(),supervised_answers=len(targets),ce=cv,aux=av,main_gradient_norm=mg,indexer_gradient_norm=ig,main_lr=opt.param_groups[0]['lr'],indexer_lr=iopt.param_groups[0]['lr'] if iopt else None,indexer_update=bool(iopt),step_wall_seconds=seconds,**bc)
                route.collect_aux=False;tick=time.perf_counter();dev=evaluate(m,development,'cuda');dev_secs=time.perf_counter()-tick;sched.step();tick=time.perf_counter();checkpoint_save(out/'checkpoint.pt',ck(True))
                if epoch+1 in [4,8,16,19]:shutil.copy2(out/'checkpoint.pt',out/f'checkpoint_epoch{epoch+1:03d}.pt')
                row=dict(epoch=epoch+1,global_step=steps,train_nll=ce_sum/1600000,train_aux=aux_sum/100000,train_accuracy=correct/1600000,development_accuracy=dev['accuracy'],development_nll=dev['nll'],train_step_seconds=epoch_secs,development_seconds=dev_secs,checkpoint_seconds=time.perf_counter()-tick);curves.append(row);event('epoch',**row);save(out/'progress.json',dict(utc=now(),status='running',**row));save(out.parent/'progress.json',dict(utc=now(),status='fitting',active_method=method,**row));print(json.dumps(dict(method=method,**row)),flush=True)
            fit=dict(status='fit_complete',method=method,started_utc=started,finished_utc=now(),epochs=19,main_updates=steps,indexer_updates=steps if iopt else 0,input_tokens=19*25600000,supervised_answers=19*1600000,wall_seconds=time.perf_counter()-timer,train_step_seconds=seconds_total,score_accounting=cost,curves=curves,final_backbone_sha=weights_sha(m.state_dict()),final_indexer_sha=weights_sha(ix.state_dict()),development_gate=curves[-1]['development_accuracy']>=.99);save(out/'fit-result.json',fit);event('fit_complete',global_step=steps)
        except Exception:
            checkpoint_save(out/'failure-checkpoint.pt',ck(False));save(out/'FAILURE.json',dict(utc=now(),steps=steps,traceback=traceback.format_exc()));event('failed',global_step=steps);raise
        finally:route.restore()
    return fit

def main(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    src=ROOT/'results/router-author-control-v0';out=args.output;out.mkdir(exist_ok=False);timer=time.perf_counter();started=now();pre=json.loads((ROOT/'results/router-author-sparse-preflight-v0/result.json').read_text());assert pre['scientific_launch_allowed'];old=json.loads((src/'result.json').read_text());assert old['runs'][1]['gate'] and old['runs'][1]['epochs']==19
    cfg=author_configs()[1];d=torch.load(src/'data.pt',map_location='cpu',weights_only=True);initial=torch.load(src/'initialization.pt',map_location='cpu',weights_only=True);ixstate=initial_indexers();torch.save(dict(model=initial,indexers=ixstate),out/'initialization.pt')
    snap=out/'source';snap.mkdir()
    for name in ['run_router_author_sparse.py','router_author_sparse.py','preflight_router_author_sparse.py','router_author_control.py','run_router_author_control.py','router_recipe_baselines.py','router_pressure_models.py','run_joint_token_router.py','run_frozen_router.py','joint_token_routing.py','frozen_routing.py','zoology_entry.py','zoology_sparse_schedule.py']:shutil.copy2(ROOT/'src'/name,snap/name)
    shutil.copytree(UPSTREAM,snap/'upstream',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));shutil.copy2(ROOT/'docs/router-author-sparse-plan-2026-09-14.md',out/'plan.md');shutil.copy2(ROOT/'results/router-author-sparse-preflight-v0/result.json',out/'preflight.json');save(out/'config.json',cfg)
    test=generated(2026091622,1024);validate(test);new_hashes=row_hashes(test)
    for key in ['train','development','test']:assert not new_hashes&row_hashes(d[key])
    assert len(new_hashes)==1024;swapped,noisy,pairs=interventions(test);mask=test['inputs']==0;fill=torch.randint(0,8192,test['inputs'].shape,generator=torch.Generator().manual_seed(2026091623));noisy['inputs']=test['inputs'].clone();noisy['inputs'][mask]=fill[mask];validate(noisy,False);fresh=dict(test=test,swapped=swapped,noisy=noisy,swap_positions=pairs);torch.save(fresh,out/'evaluation-data.pt')
    save(out/'provenance.json',dict(utc=started,old_result_sha256=sha(src/'result.json'),training_data_path=src.relative_to(ROOT).as_posix()+'/data.pt',training_data_sha256=sha(src/'data.pt'),dense_checkpoint_path=src.relative_to(ROOT).as_posix()+'/lr2/checkpoint.pt',dense_checkpoint_sha256=sha(src/'lr2/checkpoint.pt'),plan_sha256=sha(out/'plan.md'),evaluation_data_sha256=sha(out/'evaluation-data.pt'),initial_backbone_sha=weights_sha(initial),initial_indexer_sha=weights_sha(ixstate),test_seed=2026091622,noise_seed=2026091623,all_test_rows_new=True,main_parameters=1214720,indexer_parameters=8256))
    save(out/'environment.json',dict(utc=started,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),numpy=np.__version__,python=sys.version,tf32=False))
    runs=[]
    for method in METHODS:
        save(out/'progress.json',dict(utc=now(),status='fitting',active_method=method,completed_methods=[r['method'] for r in runs]));runs.append(run_one(out/method,method,cfg,initial,ixstate,d['train'],d['development'],timer))
    # Neither old nor fresh test results choose any step count or checkpoint in these fits.
    dense_ck=torch.load(src/'lr2/checkpoint.pt',map_location='cpu',weights_only=False);dense=make_model(cfg,dense_ck['model'],'cuda');dense_result=dict(method='dense_existing',new_training_updates=0,original_updates=7429,epochs=19,original_result_sha256=sha(src/'lr2/result.json'))
    for key in ['test','swapped','noisy']:dense_result[key]=evaluate(dense,fresh[key],'cuda')
    assert weights_sha(dense.state_dict())==old['runs'][1]['final_backbone_sha'];valid=all(dense_result[key]['accuracy']>=.99 for key in ['test','swapped']);save(out/'dense-reference.json',dense_result)
    for r in runs:
        ck=torch.load(out/r['method']/'checkpoint.pt',map_location='cpu',weights_only=False);m,ix,route=build(cfg,ck['model'],ck['indexers'],r['method'],'cuda');route.epoch=19;route.collect_aux=False
        for key in ['test','swapped','noisy']:r[key]=evaluate(m,fresh[key],'cuda')
        route.restore();assert weights_sha(m.state_dict())==r['final_backbone_sha'] and weights_sha(ix.state_dict())==r['final_indexer_sha'];r['gate']=valid and r['development_gate'] and all(r[key]['accuracy']>=.99 and r[key]['accuracy']>=dense_result[key]['accuracy']-.01 for key in ['test','swapped']);r['status']='complete';save(out/r['method']/'result.json',r)
    result=dict(status='complete',started_utc=started,finished_utc=now(),wall_seconds=time.perf_counter()-timer,control_valid=valid,dense_reference=dense_result,runs=runs,main_updates=sum(r['main_updates'] for r in runs),indexer_updates=sum(r['indexer_updates'] for r in runs),input_tokens=sum(r['input_tokens'] for r in runs),supervised_answers=sum(r['supervised_answers'] for r in runs),technical_main_updates=32,technical_indexer_updates=24,scope='Known recipe pilot; dense-selected 19 epoch budget and main learning rate; one seed, synthetic prefix layout, not separately tuned, no originality or speedup claim.')
    save(out/'result.json',result);save(out/'SUCCESS.json',dict(utc=now()));save(out/'progress.json',dict(utc=now(),status='complete',control_valid=valid));save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(status='complete',control_valid=valid,runs=[dict(method=r['method'],test=r['test']['accuracy'],swapped=r['swapped']['accuracy'],noisy=r['noisy']['accuracy'],gate=r['gate']) for r in runs])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    def interrupted(sig,frame):raise RuntimeError('Signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:main(args)
    except Exception:
        if args.output.exists():save(args.output/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()))
        raise
