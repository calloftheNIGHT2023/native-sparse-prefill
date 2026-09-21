"""Gated 40/80-epoch KV16 pressure study. Plans, updates, timing, and test use are explicit."""
import argparse,json,math,os,random,shutil,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from zoology_entry import ROOT,UPSTREAM,LanguageModel,set_determinism
from zoology.config import ModelConfig
from frozen_routing import TokenIndexer
from router_recipe_baselines import RECIPES
from router_pressure_models import METHODS,build,load_checkpoint
from run_frozen_router import now,save,sha,weights_sha,evaluate

def atomic_checkpoint(path,ck):
    tmp=path.with_suffix('.tmp');torch.save(ck,tmp);os.replace(tmp,path)

def initial_states(base):
    set_determinism(123);m=LanguageModel(ModelConfig(**base['model']));assert sum(p.numel() for p in m.parameters())==445952
    set_determinism(2026091470);ix=torch.nn.ModuleList([TokenIndexer(128,16) for _ in range(2)])
    return m.state_dict(),ix.state_dict()

def validate_data(folder):
    for r in json.loads((folder/'manifest.json').read_text()):assert sha(folder/r['path'])==r['sha256']
    data=torch.load(folder/'data.pt',map_location='cpu',weights_only=True);hashes=[]
    for name,count in [('train',10000),('development',1000),('test',1024)]:
        x=data[name]['inputs'];y=data[name]['labels'];assert x.shape==y.shape==(count,128);seen=set()
        import hashlib
        for xx,yy in zip(x,y):
            bank={int(xx[i]):int(xx[i+1]) for i in range(0,32,2)};p=torch.where(yy!=-100)[0];assert len(bank)==len(set(bank.values()))==16 and len(p)==16 and int(p.min())>=32
            for j in p:assert bank[int(xx[j])]==int(yy[j])
            h=hashlib.sha256(xx.numpy().tobytes()).hexdigest();assert h not in seen;seen.add(h)
        for previous in hashes:assert not previous&seen
        hashes.append(seen)
    for i,(x,y) in enumerate(zip(data['swapped']['inputs'],data['swapped']['labels'])):
        pos=torch.where(y!=-100)[0];assert len(pos)==1;bank={int(x[j]):int(x[j+1]) for j in range(0,32,2)};assert bank[int(x[pos[0]])]==int(y[pos[0]])
        assert int((x!=data['test']['inputs'][i]).sum())==2 and torch.equal(x[32:],data['test']['inputs'][i,32:])
    return data

def preflight(base,initial,initial_ix,x,device):
    records=[]
    for method in METHODS:
        a,ai,ar=build(base,initial,initial_ix,method,'cpu');b,bi,br=build(base,initial,initial_ix,method,device);a.eval();b.eval();ar.epoch=br.epoch=5;ar.collect_aux=br.collect_aux=False
        oa=a(x);ob=b(x.to(device));torch.testing.assert_close(oa,ob.cpu(),rtol=3e-4,atol=3e-5);error=float((oa-ob.cpu()).detach().abs().max())
        b.train();br.collect_aux=True;br.reset();logits=b(x.to(device));loss=logits.square().mean()+sum(br.losses);loss.backward();assert all(torch.isfinite(p.grad).all() for p in list(b.parameters())+list(bi.parameters()) if p.grad is not None)
        records.append(dict(method=method,forward_max_error=error,finite_gradients=True,optimizer_updates=0));ar.restore();br.restore()
    return records

def run_one(out,method,base,initial,initial_ix,data,device,common_epochs,overall_timer):
    out.mkdir(exist_ok=False);started=now();timer=time.perf_counter();m,ix,route=build(base,initial,initial_ix,method,device)
    assert weights_sha(m.state_dict())==weights_sha(initial)
    if method in RECIPES:assert weights_sha(ix.state_dict())==weights_sha(initial_ix)
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.1);iopt=torch.optim.AdamW(ix.parameters(),lr=.001,weight_decay=0) if method in RECIPES else None;sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=0.);set_determinism(123)
    train={k:v.to(device) for k,v in data['train'].items()};steps=0;epoch=-1;target=40 if method=='dense' else common_epochs;curves=[];cost={k:0 for k in ['main_score_entries','index_score_entries','teacher_distribution_entries']};step_seconds=0.;dev_seconds=0.;save_seconds=0.
    save(out/'config.json',dict(method=method,base=base,data_spec=dict(length=128,num_kv_pairs=16,train_rows=10000,development_rows=1000,batch_size=32,seed=123,indexer_seed=2026091470),initial_backbone_sha=weights_sha(initial),initial_indexer_sha=weights_sha(initial_ix) if iopt else None,main_parameters=445952,indexer_parameters=sum(p.numel() for p in ix.parameters()),target_epochs=target,dense_budget_rule='40 then 80 only if dense development at 40 is below 0.99'))
    def ck(boundary):
        return dict(method=method,config=base,model=m.state_dict(),indexers=ix.state_dict(),main_optimizer=opt.state_dict(),indexer_optimizer=iopt.state_dict() if iopt else None,scheduler=sched.state_dict(),epoch=epoch,steps=steps,epoch_boundary=boundary,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None,numpy_rng=np.random.get_state(),python_rng=random.getstate())
    with (out/'events.jsonl').open('w',encoding='utf-8',buffering=65536) as log:
        def event(kind,**payload):
            log.write(json.dumps(dict(utc=now(),event=kind,elapsed_seconds=time.perf_counter()-timer,**payload))+'\n')
            if kind!='optimizer_step':log.flush()
        event('start',method=method)
        try:
            while epoch+1<target:
                epoch+=1;route.epoch=epoch;m.train();route.collect_aux=True;ce_sum=0.;aux_sum=0.;epoch_steps_seconds=0.
                for first in range(0,len(train['inputs']),32):
                    if time.perf_counter()-timer>1200 or time.perf_counter()-overall_timer>5400:raise TimeoutError('Pressure-study fixed wall cap')
                    tick=time.perf_counter();opt.zero_grad()
                    if iopt:iopt.zero_grad()
                    route.reset();x=train['inputs'][first:first+32];y=train['labels'][first:first+32];logits=m(x);ce=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten());aux=sum(route.losses);(ce+aux).backward()
                    mg=float(torch.stack([p.grad.square().sum() for p in m.parameters() if p.grad is not None]).sum().sqrt());ig=float(torch.stack([p.grad.square().sum() for p in ix.parameters() if p.grad is not None]).sum().sqrt()) if iopt else 0.;cv=float(ce.detach());av=float(aux.detach()) if isinstance(aux,torch.Tensor) else float(aux)
                    assert all(math.isfinite(z) for z in (cv,av,mg,ig));opt.step()
                    if iopt:iopt.step()
                    if device=='cuda':torch.cuda.synchronize()
                    elapsed=time.perf_counter()-tick;step_seconds+=elapsed;epoch_steps_seconds+=elapsed;steps+=1;ce_sum+=cv*len(x);aux_sum+=av*len(x);batch_cost={k:sum(c[k] for c in route.cost) for k in cost}
                    for k,v in batch_cost.items():cost[k]+=v
                    event('optimizer_step',global_step=steps,epoch=epoch+1,batch_examples=len(x),supervised_answers=int((y!=-100).sum()),input_tokens=x.numel(),ce=cv,aux=av,main_gradient_norm=mg,indexer_gradient_norm=ig,main_lr=opt.param_groups[0]['lr'],indexer_update=bool(iopt),step_wall_seconds=elapsed,**batch_cost)
                m.eval();route.collect_aux=False;tick=time.perf_counter();dev=evaluate(m,data['development'],device);dev_time=time.perf_counter()-tick;dev_seconds+=dev_time;sched.step()
                row=dict(epoch=epoch+1,global_step=steps,train_nll=ce_sum/10000,train_aux=aux_sum/10000,development_accuracy=dev['accuracy'],development_nll=dev['nll'],train_step_seconds=epoch_steps_seconds,development_seconds=dev_time)
                tick=time.perf_counter();atomic_checkpoint(out/'checkpoint.pt',ck(True))
                if method=='dense' and epoch+1 in (40,80):shutil.copy2(out/'checkpoint.pt',out/f'checkpoint_epoch{epoch+1:03d}.pt')
                save_time=time.perf_counter()-tick;save_seconds+=save_time;row['checkpoint_seconds']=save_time;curves.append(row);event('epoch',**row);print(json.dumps(dict(method=method,**row)),flush=True);save(out/'progress.json',dict(utc=now(),status='running',**row))
                if method=='dense' and epoch==39:
                    target=40 if dev['accuracy']>=.99 else 80;decision=dict(utc=now(),development_accuracy=dev['accuracy'],common_epochs=target,test_consulted=False);save(out/'budget-decision.json',decision);event('budget_decision',common_epochs=target,development_accuracy=dev['accuracy'])
            result=dict(status='fit_complete',method=method,started_utc=started,finished_utc=now(),epochs=target,main_updates=steps,indexer_updates=steps if iopt else 0,input_tokens=target*10000*128,supervised_answers=target*10000*16,wall_seconds=time.perf_counter()-timer,train_step_seconds=step_seconds,development_seconds=dev_seconds,checkpoint_seconds=save_seconds,score_accounting=cost,curves=curves,final_backbone_sha=weights_sha(m.state_dict()),final_indexer_sha=weights_sha(ix.state_dict()),development_gate=curves[-1]['development_accuracy']>=.99)
            save(out/'fit-result.json',result);event('complete',global_step=steps)
        except Exception:
            atomic_checkpoint(out/'failure-checkpoint.pt',ck(False));save(out/'FAILURE.json',dict(utc=now(),steps=steps,traceback=traceback.format_exc()));event('failed',global_step=steps);raise
        finally:route.restore()
    return result

def main(args):
    out=args.output;out.mkdir(exist_ok=False);started=now();timer=time.perf_counter();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data_folder=ROOT/'data/router-pressure-kv16-v0';data=validate_data(data_folder);base=json.loads((data_folder/'config.json').read_text());initial,initial_ix=initial_states(base)
    torch.save(dict(model=initial,indexers=initial_ix),out/'initialization.pt');snap=out/'source';snap.mkdir()
    for name in ['run_router_pressure.py','router_pressure_models.py','router_recipe_baselines.py','run_joint_token_router.py','run_frozen_router.py','frozen_routing.py','joint_token_routing.py','zoology_entry.py','zoology_sparse_schedule.py']:shutil.copy2(ROOT/'src'/name,snap/name)
    shutil.copytree(UPSTREAM,snap/'upstream',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));shutil.copy2(ROOT/'docs/router-pressure-plan-2026-09-14.md',out/'preregistered-plan.md');shutil.copy2(data_folder/'audit.json',out/'source-data-audit.json')
    save(out/'environment.json',dict(utc=started,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,device=args.device,torch=str(torch.__version__),python=sys.version,data_sha256=sha(data_folder/'data.pt'),plan_sha256=sha(out/'preregistered-plan.md')));save(out/'preflight.json',preflight(base,initial,initial_ix,data['train']['inputs'][:2],args.device))
    runs=[run_one(out/'dense','dense',base,initial,initial_ix,data,args.device,None,timer)];common=runs[0]['epochs'];allowed=runs[0]['development_gate'];save(out/'progress.json',dict(utc=now(),status='fitting' if allowed else 'control_failed',common_epochs=common,completed=['dense']))
    if allowed:
        for method in METHODS[1:]:
            runs.append(run_one(out/method,method,base,initial,initial_ix,data,args.device,common,timer));save(out/'progress.json',dict(utc=now(),status='fitting',common_epochs=common,completed=[r['method'] for r in runs]))
    # No held-out evaluation occurs before every admitted fit is complete.
    for r in runs:
        ck=torch.load(out/r['method']/'checkpoint.pt',map_location='cpu',weights_only=False);m,ix,route=load_checkpoint(ck,args.device)
        for key in ['test','swapped']:r[key]=evaluate(m,data[key],args.device)
        route.restore();assert weights_sha(m.state_dict())==r['final_backbone_sha'] and weights_sha(ix.state_dict())==r['final_indexer_sha'];r['status']='complete';save(out/r['method']/'result.json',r)
    valid=allowed and all(runs[0][k]['accuracy']>=.99 for k in ['test','swapped'])
    for r in runs[1:]:r['gate']=valid and all(r[k]['accuracy']>=.99 and r[k]['accuracy']>=runs[0][k]['accuracy']-.01 for k in ['test','swapped']);save(out/r['method']/'result.json',r)
    result=dict(status='complete' if allowed else 'complete_control_failed',started_utc=started,finished_utc=now(),wall_seconds=time.perf_counter()-timer,common_epochs=common,control_valid=valid,skipped_methods=[] if allowed else list(METHODS[1:]),main_updates=sum(r['main_updates'] for r in runs),indexer_updates=sum(r['indexer_updates'] for r in runs),input_tokens=sum(r['input_tokens'] for r in runs),supervised_answers=sum(r['supervised_answers'] for r in runs),runs=runs,scope='Single-seed synthetic KV16 pressure control. Known recipes, no method novelty, no real-text or Qwen4 result.')
    save(out/'result.json',result);save(out/'SUCCESS.json',dict(utc=now(),status=result['status']));save(out/'progress.json',dict(utc=now(),status=result['status']));save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(status=result['status'],common_epochs=common,control_valid=valid,runs=[dict(method=r['method'],test=r['test']['accuracy'],swapped=r['swapped']['accuracy'],gate=r.get('gate')) for r in runs])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    def interrupted(sig,frame):raise RuntimeError('Signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:main(a)
    except Exception:
        if a.output.exists():save(a.output/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()))
        raise
