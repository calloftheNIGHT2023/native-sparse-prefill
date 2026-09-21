"""Bounded reproduction of two preselected author learning rates, stopping on development."""
import argparse,json,math,os,random,shutil,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from zoology_entry import ROOT,UPSTREAM,set_determinism
from router_author_control import SOURCE,LRS,author_configs,make_model,generated,validate,row_hashes,interventions,selected_logits,evaluate
from run_frozen_router import now,save,sha,weights_sha

def checkpoint_save(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)

def run_one(folder,cfg,initial,data,device,batch_timer):
    folder.mkdir(exist_ok=False);started=now();timer=time.perf_counter();m=make_model(cfg,initial,device);assert weights_sha(m.state_dict())==weights_sha(initial)
    opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=64,eta_min=0.);set_determinism(123)
    train={k:v.to(device) for k,v in data['train'].items()};steps=0;epoch=-1;curves=[];train_seconds=0.;save(folder/'config.json',dict(config=cfg,initial_backbone_sha=weights_sha(initial),parameters=sum(p.numel() for p in m.parameters()),head_implementation='project_only_supervised_positions'))
    def ck(boundary):return dict(config=cfg,model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,steps=steps,epoch_boundary=boundary,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    with (folder/'events.jsonl').open('w',encoding='utf-8',buffering=65536) as f:
        def event(kind,**kw):
            f.write(json.dumps(dict(utc=now(),event=kind,elapsed_seconds=time.perf_counter()-timer,**kw))+'\n')
            if kind!='optimizer_step':f.flush()
        event('start',learning_rate=cfg['learning_rate'])
        try:
            for epoch in range(64):
                m.train();ce_sum=0.;correct=0;epoch_secs=0.
                for first in range(0,100000,256):
                    if time.perf_counter()-timer>2700 or time.perf_counter()-batch_timer>5400:raise TimeoutError('Author-control fixed wall cap')
                    tick=time.perf_counter();opt.zero_grad();x=train['inputs'][first:first+256];y=train['labels'][first:first+256];logits,targets=selected_logits(m,x,y);ce=torch.nn.functional.cross_entropy(logits,targets);ce.backward();grad=float(torch.stack([p.grad.square().sum() for p in m.parameters() if p.grad is not None]).sum().sqrt());loss=float(ce.detach());assert math.isfinite(loss) and math.isfinite(grad);opt.step();torch.cuda.synchronize();seconds=time.perf_counter()-tick;epoch_secs+=seconds;train_seconds+=seconds;steps+=1;ce_sum+=loss*len(targets);correct+=int((logits.detach().argmax(-1)==targets).sum())
                    event('optimizer_step',global_step=steps,epoch=epoch+1,batch_examples=len(x),input_tokens=x.numel(),supervised_answers=len(targets),loss=loss,gradient_norm=grad,learning_rate=opt.param_groups[0]['lr'],train_step_seconds=seconds)
                tick=time.perf_counter();dev=evaluate(m,data['development'],device);dev_sec=time.perf_counter()-tick;passed=dev['accuracy']>.99
                # Author Trainer.fit checks its stopping threshold before scheduler.step.
                if not passed:sched.step()
                tick=time.perf_counter();checkpoint_save(folder/'checkpoint.pt',ck(True))
                if epoch+1 in (8,16,32,64):shutil.copy2(folder/'checkpoint.pt',folder/f'checkpoint_epoch{epoch+1:03d}.pt')
                row=dict(epoch=epoch+1,global_step=steps,train_nll=ce_sum/1600000,train_accuracy=correct/1600000,development_accuracy=dev['accuracy'],development_nll=dev['nll'],train_step_seconds=epoch_secs,development_seconds=dev_sec,checkpoint_seconds=time.perf_counter()-tick,early_stop=passed,scheduler_last_epoch=sched.last_epoch);curves.append(row);event('epoch',**row);save(folder/'progress.json',dict(utc=now(),status='running',**row));print(json.dumps(dict(learning_rate=cfg['learning_rate'],**row)),flush=True)
                if passed:break
            fit=dict(status='fit_complete',method=folder.name,learning_rate=cfg['learning_rate'],started_utc=started,finished_utc=now(),epochs=epoch+1,main_updates=steps,indexer_updates=0,input_tokens=(epoch+1)*25600000,supervised_answers=(epoch+1)*1600000,train_step_seconds=train_seconds,wall_seconds=time.perf_counter()-timer,development_gate=curves[-1]['development_accuracy']>.99,curves=curves,final_backbone_sha=weights_sha(m.state_dict()));save(folder/'fit-result.json',fit);event('fit_complete',global_step=steps)
        except Exception:
            checkpoint_save(folder/'failure-checkpoint.pt',ck(False));save(folder/'FAILURE.json',dict(utc=now(),steps=steps,traceback=traceback.format_exc()));event('failed',global_step=steps);raise
    return fit

def main(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.output;out.mkdir(exist_ok=False);timer=time.perf_counter();started=now();preflight=json.loads((ROOT/'results/router-author-preflight-v0/result.json').read_text());assert preflight['scientific_launch_allowed'];shutil.copy2(ROOT/'results/router-author-preflight-v0/result.json',out/'preflight.json')
    source=out/'source';source.mkdir()
    for name in ['run_router_author_control.py','router_author_control.py','preflight_router_author.py','zoology_entry.py','run_frozen_router.py','frozen_routing.py','zoology_sparse_schedule.py']:shutil.copy2(ROOT/'src'/name,source/name)
    shutil.copytree(UPSTREAM,source/'upstream',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));shutil.copy2(SOURCE,source/'author-paper-configs.py');shutil.copy2(ROOT/'docs/router-author-control-plan-2026-09-14.md',out/'plan.md')
    cfgs=author_configs();save(out/'resolved-configs.json',cfgs);set_determinism(123);init_model=make_model(cfgs[0]);initial=init_model.state_dict();torch.save(initial,out/'initialization.pt');parameters=sum(p.numel() for p in init_model.parameters());assert parameters==1214720
    rng=np.random.RandomState(123);train_seed=int(rng.randint(0,2**31,size=1,dtype=np.int64)[0]);dev_seed=int(rng.randint(2**31,2**32,size=1,dtype=np.int64)[0]);seeds=dict(train=train_seed,development=dev_seed,test=2026091601,paired_noise=2026091602);save(out/'data-seeds.json',seeds)
    save(out/'progress.json',dict(utc=now(),status='generating_data',scientific_updates=0));save(out/'environment.json',dict(utc=started,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),numpy=np.__version__,python=sys.version,plan_sha256=sha(out/'plan.md'),author_source_sha256=sha(SOURCE),parameters=parameters))
    data={};seen=set();audit=[]
    for key,rows in [('train',100000),('development',3000),('test',1024)]:
        data[key]=generated(seeds[key],rows);validate(data[key]);hashes=row_hashes(data[key]);assert len(hashes)==rows and not hashes&seen;seen.update(hashes);audit.append(dict(split=key,rows=rows,answers=16*rows,labels_verified=True,disjoint=True));print(json.dumps(dict(event='data_ready',split=key,rows=rows)),flush=True)
    data['swapped'],data['noisy'],pairs=interventions(data['test']);data['swap_positions']=pairs;torch.save(data,out/'data.pt');save(out/'data-audit.json',dict(utc=now(),splits=audit,seeds=seeds,data_sha256=sha(out/'data.pt'),initial_backbone_sha=weights_sha(initial)))
    runs=[]
    for i,cfg in enumerate(cfgs):
        run=run_one(out/f'lr{i+1}',cfg,initial,data,'cuda',timer);runs.append(run);save(out/'progress.json',dict(utc=now(),status='fitting',completed=[r['method'] for r in runs]))
        if run['development_gate']:break
    # Tests are evaluated only after all admitted fitting has finished.
    for run in runs:
        ck=torch.load(out/run['method']/'checkpoint.pt',map_location='cpu',weights_only=False);m=make_model(ck['config'],ck['model'],'cuda')
        for key in ['test','swapped','noisy']:run[key]=evaluate(m,data[key],'cuda')
        assert weights_sha(m.state_dict())==run['final_backbone_sha'];run['gate']=run['development_gate'] and run['test']['accuracy']>=.99 and run['swapped']['accuracy']>=.99;run['status']='complete';save(out/run['method']/'result.json',run)
    result=dict(status='complete',started_utc=started,finished_utc=now(),wall_seconds=time.perf_counter()-timer,main_updates=sum(r['main_updates'] for r in runs),indexer_updates=0,input_tokens=sum(r['input_tokens'] for r in runs),supervised_answers=sum(r['supervised_answers'] for r in runs),technical_main_updates=8,runs=runs,skipped_learning_rates=list(LRS[len(runs):]),control_valid=any(r['gate'] for r in runs),scope='Single-seed author-configuration subset, dense attention, supervised-position head verified equivalent; no sparse algorithm or paired attribution of earlier failures.')
    save(out/'result.json',result);save(out/'SUCCESS.json',dict(utc=now()));save(out/'progress.json',dict(utc=now(),status='complete',control_valid=result['control_valid']));save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(status='complete',control_valid=result['control_valid'],runs=[dict(learning_rate=r['learning_rate'],epochs=r['epochs'],test=r['test']['accuracy'],swapped=r['swapped']['accuracy'],noisy=r['noisy']['accuracy']) for r in runs])),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    def interrupted(sig,frame):raise RuntimeError('Signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:main(args)
    except Exception:
        if args.output.exists():save(args.output/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()))
        raise
