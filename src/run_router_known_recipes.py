"""Fixed-budget known-recipe adapters on MQAR; old full-KL checkpoint is reused as reference."""
import argparse,hashlib,json,math,random,shutil,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from zoology_entry import ROOT,UPSTREAM,set_determinism,multiquery_ar
from run_joint_token_router import make_model
from resume_joint_token_router import restore_checkpoint
from frozen_routing import TokenIndexer
from router_recipe_baselines import RecipeRouter,RECIPES
from run_frozen_router import save,sha,now,weights_sha,evaluate,value_swap

def build(ck,device):
    m=make_model(ck['config'],ck['model'],device);ix=torch.nn.ModuleList([TokenIndexer(128,16) for _ in range(2)]).to(device);ix.load_state_dict(ck['indexers']);r=RecipeRouter(m,ix,ck['recipe']);r.epoch=ck['epoch']+1;r.collect_aux=False;m.eval();return m,ix,r

def preflight(device):
    source=ROOT/'results/joint-token-router-v0';ck=torch.load(source/'learned_r16/checkpoint.pt',map_location='cpu',weights_only=False)
    x=torch.randint(0,256,(2,16));out=[]
    for recipe in RECIPES:
        c=dict(ck,recipe=recipe);cpu,_,cr=build(c,'cpu');gpu,gix,gr=build(c,device);a=cpu(x);b=gpu(x.to(device));torch.testing.assert_close(a,b.cpu(),rtol=3e-4,atol=3e-5)
        gpu.train();gr.collect_aux=True;gr.reset();logits=gpu(x.to(device))
        if recipe=='warm4_selectedkl_r16':
            sum(gr.losses).backward();assert all(p.grad is None for p in gpu.parameters());assert any(p.grad is not None and p.grad.abs().sum()>0 for p in gix.parameters())
        else:
            logits.square().mean().backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in gix.parameters())
        assert all(torch.isfinite(p.grad).all() for p in list(gpu.parameters())+list(gix.parameters()) if p.grad is not None)
        out.append(dict(recipe=recipe,max_error=float((a-b.cpu()).abs().max()),optimizer_updates=0));cr.restore();gr.restore()
    return out

def run_one(folder,recipe,base,initial,initial_ix,data,device,overall_timer):
    folder.mkdir(exist_ok=False);started=now();timer=time.perf_counter();set_determinism(123);m=make_model(base,initial,device);set_determinism(2026091470);ix=torch.nn.ModuleList([TokenIndexer(128,16) for _ in range(2)]).to(device);ix.load_state_dict(initial_ix);route=RecipeRouter(m,ix,recipe)
    assert weights_sha(m.state_dict())==weights_sha(initial) and weights_sha(ix.state_dict())==weights_sha(initial_ix)
    opt=torch.optim.AdamW(m.parameters(),lr=base['learning_rate'],weight_decay=base['weight_decay']);iopt=torch.optim.AdamW(ix.parameters(),lr=.001,weight_decay=0);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=0.);set_determinism(123)
    train={k:v.to(device) for k,v in data['train'].items()};steps=0;epoch=-1;curves=[];cost=dict(main_score_entries=0,index_score_entries=0,teacher_distribution_entries=0)
    save(folder/'config.json',dict(recipe=recipe,base=base,epochs=40,batch=32,seed=123,indexer_seed=2026091470,initial_backbone_sha=weights_sha(initial),initial_indexer_sha=weights_sha(initial_ix),scope='Known recipe token-level adaptation; not original MSA/KSA model reproduction'))
    def checkpoint(boundary):
        return dict(model=m.state_dict(),indexers=ix.state_dict(),main_optimizer=opt.state_dict(),indexer_optimizer=iopt.state_dict(),scheduler=scheduler.state_dict(),steps=steps,epoch=epoch,epoch_boundary=boundary,rank=16,config=base,recipe=recipe,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None,numpy_rng=np.random.get_state(),python_rng=random.getstate())
    with (folder/'events.jsonl').open('w',encoding='utf-8',buffering=65536) as log:
        def event(kind,**payload):
            log.write(json.dumps(dict(utc=now(),event=kind,elapsed_seconds=time.perf_counter()-timer,**payload))+'\n')
            if kind!='optimizer_step':log.flush()
        event('start',recipe=recipe)
        try:
            for epoch in range(40):
                route.epoch=epoch;m.train();route.collect_aux=True;ce_sum=0.;aux_sum=0.
                for first in range(0,len(train['inputs']),32):
                    if time.perf_counter()-timer>720 or time.perf_counter()-overall_timer>1800:raise TimeoutError('Preregistered wall cap')
                    x=train['inputs'][first:first+32];y=train['labels'][first:first+32];opt.zero_grad();iopt.zero_grad();route.reset();logits=m(x);ce=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten());aux=sum(route.losses);(ce+aux).backward()
                    mg=float(torch.stack([p.grad.square().sum() for p in m.parameters() if p.grad is not None]).sum().sqrt());ig=float(torch.stack([p.grad.square().sum() for p in ix.parameters() if p.grad is not None]).sum().sqrt());cv=float(ce.detach());av=float(aux.detach()) if isinstance(aux,torch.Tensor) else float(aux)
                    assert all(math.isfinite(z) for z in [mg,ig,cv,av]);opt.step();iopt.step();steps+=1;ce_sum+=cv*len(x);aux_sum+=av*len(x)
                    batch_cost={k:sum(c[k] for c in route.cost) for k in cost}
                    for k in cost:cost[k]+=batch_cost[k]
                    event('optimizer_step',global_step=steps,epoch=epoch+1,batch_examples=len(x),ce=cv,aux=av,main_gradient_norm=mg,indexer_gradient_norm=ig,main_lr=opt.param_groups[0]['lr'],indexer_lr=.001,**batch_cost)
                m.eval();route.collect_aux=False;dev=evaluate(m,data['development'],device);scheduler.step();row=dict(epoch=epoch+1,global_step=steps,train_nll=ce_sum/10000,train_aux=aux_sum/10000,development_accuracy=dev['accuracy'],development_nll=dev['nll']);curves.append(row);event('epoch',**row);print(json.dumps(dict(recipe=recipe,**row)),flush=True);torch.save(checkpoint(True),folder/'checkpoint.pt')
                save(folder/'progress.json',dict(utc=now(),status='running',**row))
            result=dict(status='fit_complete',recipe=recipe,started_utc=started,finished_utc=now(),epochs=40,main_updates=steps,indexer_updates=steps,input_tokens=25600000,supervised_answers=1600000,wall_seconds=time.perf_counter()-timer,curves=curves,score_accounting=cost,final_backbone_sha=weights_sha(m.state_dict()),final_indexer_sha=weights_sha(ix.state_dict()))
            save(folder/'fit-result.json',result);event('complete',global_step=steps)
        except Exception:
            torch.save(checkpoint(False),folder/'failure-checkpoint.pt');save(folder/'FAILURE.json',dict(utc=now(),steps=steps,traceback=traceback.format_exc()));event('failed',global_step=steps);raise
        finally:route.restore()
    return result

def main(args):
    out=args.output;out.mkdir(exist_ok=False);timer=time.perf_counter();start=now();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'results/joint-token-router-v0';data=torch.load(source/'data-and-initialization.pt',map_location='cpu',weights_only=True);ck=torch.load(source/'learned_r16/checkpoint.pt',map_location='cpu',weights_only=False);base=ck['config'];initial_ix=torch.load(source/'learned_r16/initial-indexer.pt',map_location='cpu',weights_only=True)
    snap=out/'source';snap.mkdir()
    for name in ['run_router_known_recipes.py','router_recipe_baselines.py','resume_joint_token_router.py','run_joint_token_router.py','run_frozen_router.py','frozen_routing.py','joint_token_routing.py','zoology_entry.py','zoology_sparse_schedule.py']:shutil.copy2(ROOT/'src'/name,snap/name)
    shutil.copytree(UPSTREAM,snap/'upstream',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));shutil.copy2(ROOT/'docs/router-known-recipes-plan-2026-09-14.md',out/'preregistered-plan.md')
    torch.manual_seed(2026091492);d=multiquery_ar(vocab_size=256,num_examples=1024,input_seq_len=64,seed=2026091492,num_kv_pairs=4);fresh=dict(inputs=d.inputs,labels=d.labels);swapped,pairs=value_swap(fresh)
    previous=[data[k]['inputs'] for k in ['train','development','fresh','swapped']]
    for path in ['results/frozen-router-capacity-v0/data.pt','results/joint-token-router-epoch80-v0/new-test.pt']:
        old=torch.load(ROOT/path,map_location='cpu',weights_only=True);previous.extend([old[k]['inputs'] for k in ['fresh','swapped']])
    hist=torch.load(ROOT/'results/schedule-screen-cloud-v0/shared-data-and-initialization.pt',map_location='cpu',weights_only=True);previous.extend(v['inputs'] for v in hist['splits'].values())
    rowsha=lambda x:hashlib.sha256(x.numpy().tobytes()).hexdigest();oldhash={rowsha(x) for batch in previous for x in batch};newhash={rowsha(x) for x in fresh['inputs']};assert len(newhash)==1024 and not newhash&oldhash
    for x,y in zip(fresh['inputs'],fresh['labels']):
        mapping={int(x[j]):int(x[j+1]) for j in range(0,8,2)};assert len(mapping)==4 and int((y!=-100).sum())==4
        for p in torch.where(y!=-100)[0]:assert mapping[int(x[p])]==int(y[p])
    torch.save(dict(fresh=fresh,swapped=swapped,swap_positions=pairs),out/'new-test.pt');save(out/'data-audit.json',dict(seed=2026091492,rows=1024,answers=4096,swap_answers=1024,disjoint_previous_rows=True,labels_verified=True,train_data_sha=sha(source/'data-and-initialization.pt')))
    save(out/'environment.json',dict(utc=start,device=args.device,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,torch=str(torch.__version__),python=sys.version));save(out/'preflight.json',preflight(args.device));runs=[]
    for recipe in RECIPES:
        runs.append(run_one(out/recipe,recipe,base,data['initial_state'],initial_ix,data,args.device,timer));save(out/'progress.json',dict(utc=now(),status='fitting',completed=[r['recipe'] for r in runs]))
    reference=dict(recipe='fullkl_r16_epoch40_reference',status='complete',checkpoint_sha256=sha(source/'learned_r16/checkpoint.pt'),new_training_updates=0)
    m,ix,route,*_=restore_checkpoint(ck,args.device);m.eval();route.collect_aux=False
    for key,d in [('fresh',fresh),('swapped',swapped)]:reference[key]=evaluate(m,d,args.device)
    route.restore()
    for r in runs:
        c=torch.load(out/r['recipe']/'checkpoint.pt',map_location='cpu',weights_only=False);m,ix,route=build(c,args.device)
        for key,d in [('fresh',fresh),('swapped',swapped)]:r[key]=evaluate(m,d,args.device)
        route.restore();assert weights_sha(m.state_dict())==r['final_backbone_sha'];r['status']='complete';r['gate']=all(r[k]['accuracy']>=.99 for k in ['fresh','swapped']);save(out/r['recipe']/'result.json',r)
    result=dict(status='complete',started_utc=start,finished_utc=now(),wall_seconds=time.perf_counter()-timer,additional_main_updates=25040,additional_indexer_updates=25040,additional_input_tokens=51200000,additional_supervised_answers=3200000,reference=reference,runs=runs,scope='Known recipe adaptations, matched 40 epochs and initial weights; old full-KL timing is not directly comparable; single seed toy task')
    save(out/'result.json',result);save(out/'SUCCESS.json',dict(utc=now()));save(out/'progress.json',dict(utc=now(),status='complete'));save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(status='complete',reference=reference['fresh']['accuracy'],runs=[dict(recipe=r['recipe'],fresh=r['fresh']['accuracy'],swapped=r['swapped']['accuracy'],gate=r['gate']) for r in runs])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args()
    def interrupted(sig,frame):raise RuntimeError('Signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:main(a)
    except Exception:
        if a.output.exists():save(a.output/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()))
        raise
