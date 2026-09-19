"""Resume the current MQAR run at an epoch boundary, retaining both optimizers and RNG."""
import argparse,copy,json,math,random,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from run_joint_token_router import make_model
from zoology_entry import set_determinism
from frozen_routing import TokenIndexer
from joint_token_routing import JointTokenRouter
from run_frozen_router import now,save,sha,evaluate,weights_sha

def restore_checkpoint(ck,device,restore_rng=True):
    set_determinism(123)
    model=make_model(ck['config'],ck['model'],device)
    ix=torch.nn.ModuleList([TokenIndexer(128,ck['rank']) for _ in range(2)]).to(device);ix.load_state_dict(ck['indexers'])
    route=JointTokenRouter(model,ix,ck['route'])
    opt=torch.optim.AdamW(model.parameters(),lr=ck['config']['learning_rate'],weight_decay=ck['config']['weight_decay'])
    iopt=torch.optim.AdamW(ix.parameters(),lr=.001,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=0.)
    opt.load_state_dict(ck['main_optimizer']);iopt.load_state_dict(ck['indexer_optimizer']);scheduler.load_state_dict(ck['scheduler'])
    if restore_rng:
        torch.set_rng_state(ck['torch_rng'].cpu());np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        if device=='cuda':
            if not ck.get('cuda_rng'):raise ValueError('No saved CUDA RNG; exact training continuation requires it.')
            if len(ck['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA device count differs; select the original number of visible devices first.')
            torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
    return model,ix,route,opt,iopt,scheduler

def pack(model,ix,opt,iopt,scheduler,base,epoch,steps,device):
    return dict(model=model.state_dict(),indexers=ix.state_dict(),main_optimizer=opt.state_dict(),indexer_optimizer=iopt.state_dict(),scheduler=scheduler.state_dict(),steps=steps,epoch=epoch,rank=base['rank'],route=base['route'],config=base['config'],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None,numpy_rng=np.random.get_state(),python_rng=random.getstate(),epoch_boundary=True)

def train_step(model,ix,route,opt,iopt,x,y):
    model.train();route.collect_aux=True;route.reset();opt.zero_grad();iopt.zero_grad()
    logits=model(x);ce=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten());kl=sum(route.losses);(ce+kl).backward()
    cg=float(torch.stack([p.grad.square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt())
    ig=float(torch.stack([p.grad.square().sum() for p in ix.parameters() if p.grad is not None]).sum().sqrt());ce=float(ce.detach());kl=float(kl.detach())
    if not all(math.isfinite(z) for z in (ce,kl,cg,ig)):raise FloatingPointError('Nonfinite continuation step')
    opt.step();iopt.step()
    return dict(ce=ce,kl_sum_layers=kl,main_gradient_norm=cg,indexer_gradient_norm=ig)

def main(a):
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);data=torch.load(a.data,map_location='cpu',weights_only=True)
    per_epoch=math.ceil(len(data['train']['inputs'])/32);start_epoch=ck['epoch']+1
    if ck['steps']!=per_epoch*start_epoch or ck.get('epoch_boundary') is False:raise ValueError('This entry accepts a completed epoch checkpoint only, not a partial failure checkpoint.')
    if a.total_epochs<=start_epoch:raise ValueError('total-epochs must exceed the completed epoch count.')
    if not a.plan.is_file():raise ValueError('Provide the frozen continuation plan before training.')
    a.output.mkdir(parents=True,exist_ok=False);started=now();timer=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    save(a.output/'resume-lineage.json',dict(utc=started,parent_checkpoint_sha256=sha(a.checkpoint),data_sha256=sha(a.data),plan_sha256=sha(a.plan),resume_from_completed_epochs=start_epoch,global_step=ck['steps'],total_epochs=a.total_epochs,device=a.device,torch=str(torch.__version__),python=sys.version,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,old_fresh_split_not_reevaluated=True))
    (a.output/'continuation-plan.md').write_bytes(a.plan.read_bytes());model,ix,route,opt,iopt,scheduler=restore_checkpoint(ck,a.device);steps=ck['steps'];start_steps=steps;epoch=start_epoch-1;stable=None
    train={k:v.to(a.device) for k,v in data['train'].items()}
    def stop(sig,frame):raise RuntimeError('Interrupted by signal '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (a.output/'events.jsonl').open('w',encoding='utf-8',buffering=65536) as log:
        def event(kind,**kw):
            log.write(json.dumps(dict(utc=now(),event=kind,elapsed_seconds=time.perf_counter()-timer,**kw))+'\n')
            if kind!='optimizer_step':log.flush()
        event('resume',global_step=steps,completed_epochs=start_epoch)
        try:
            for epoch in range(start_epoch,a.total_epochs):
                ce_sum=0.;kl_sum=0.;seen=0
                for first in range(0,len(train['inputs']),32):
                    if time.perf_counter()-timer>a.max_seconds:raise TimeoutError('Continuation wall-clock cap reached')
                    x=train['inputs'][first:first+32];y=train['labels'][first:first+32];metrics=train_step(model,ix,route,opt,iopt,x,y);steps+=1;ce_sum+=metrics['ce']*len(x);kl_sum+=metrics['kl_sum_layers']*len(x);seen+=len(x)
                    event('optimizer_step',global_step=steps,additional_step=steps-start_steps,epoch=epoch,batch_examples=len(x),**metrics)
                model.eval();route.collect_aux=False;route.reset();dev=evaluate(model,data['development'],a.device);scheduler.step()
                row=dict(epoch=epoch+1,global_step=steps,train_nll=ce_sum/seen,mean_kl_sum_layers=kl_sum/seen,development_accuracy=dev['accuracy'],development_nll=dev['nll']);event('epoch',**row);print(json.dumps(row),flush=True)
                stable=pack(model,ix,opt,iopt,scheduler,ck,epoch,steps,a.device)
                torch.save(stable,a.output/'checkpoint.pt');save(a.output/'progress.json',dict(utc=now(),status='running',**row))
            result=dict(status='fit_complete',utc=now(),completed_epochs=a.total_epochs,additional_backbone_updates=steps-start_steps,additional_indexer_updates=steps-start_steps,global_step=steps,wall_seconds=time.perf_counter()-timer,final_backbone_hash=weights_sha(model.state_dict()),fresh_evaluation='A separate newly frozen test set is required; old test scores not reused.')
            save(a.output/'result.json',result);event('complete',global_step=steps,completed_epochs=a.total_epochs,additional_updates=steps-start_steps)
        except Exception:
            # Mid-step snapshots are diagnostic only; retain last completed epoch separately.
            failure=pack(model,ix,opt,iopt,scheduler,ck,epoch,steps,a.device);failure['epoch_boundary']=False
            torch.save(failure,a.output/'failure-checkpoint.pt');save(a.output/'FAILURE.json',dict(utc=now(),global_step=steps,traceback=traceback.format_exc(),restart_from='checkpoint.pt if present, otherwise the parent checkpoint; failure snapshot is not an epoch boundary'))
            event('failed',global_step=steps);raise
        finally:route.restore()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--total-epochs',type=int,required=True);p.add_argument('--max-seconds',type=int,default=1200);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');main(p.parse_args())
