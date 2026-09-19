"""Preregistered frozen-backbone indexer fitting; scientific runs and diagnostics separated."""
import argparse,hashlib,json,math,random,shutil,signal,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from zoology_entry import ROOT,UPSTREAM,LanguageModel,multiquery_ar,set_determinism
from zoology.config import ModelConfig
from zoology_sparse_schedule import install
from frozen_routing import TokenIndexer,FrozenRouter,full_teacher,kl_loss

def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def weights_sha(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def evaluate(model,data,device):
    correct=0;count=0;nll=0.;preds=[];start=time.perf_counter()
    for first in range(0,len(data['inputs']),32):
        x=data['inputs'][first:first+32].to(device);y=data['labels'][first:first+32].to(device);logits=model(x);mask=y!=-100
        p=logits.argmax(-1)[mask];preds.extend(p.cpu().tolist());correct+=int((p==y[mask]).sum());count+=int(mask.sum())
        nll+=float(torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='sum'))
    return dict(accuracy=correct/count,nll=nll/count,answers=count,predictions=preds,wall_seconds=time.perf_counter()-start)

def value_swap(data):
    out=dict(inputs=data['inputs'].clone(),labels=torch.full_like(data['labels'],-100));pairs=[]
    for i,(x,y) in enumerate(zip(data['inputs'],data['labels'])):
        pos=int(torch.where(y!=-100)[0][-1]);src=next(k+1 for k in range(0,8,2) if x[k]==x[pos]);alt=((src-1+2)%8)+1
        out['inputs'][i,src]=x[alt];out['inputs'][i,alt]=x[src];out['labels'][i,pos]=x[alt]
        assert x[src]!=x[alt] and int((out['inputs'][i]!=x).sum())==2 and torch.equal(out['inputs'][i,8:],x[8:]);pairs.append([pos,src,alt])
    return out,pairs

def main(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);timer=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def interrupted(sig,frame):raise RuntimeError('Interrupted by signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def cap():
        if time.perf_counter()-timer>1200:raise TimeoutError('Fixed 20-minute experiment cap reached')
    snapshots=out/'source';snapshots.mkdir()
    for name in ['run_frozen_router.py','frozen_routing.py','zoology_entry.py','zoology_sparse_schedule.py']:shutil.copy2(ROOT/'src'/name,snapshots/name)
    shutil.copytree(UPSTREAM,snapshots/'upstream',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    checkpoint=args.source/'native/checkpoint.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=LanguageModel(ModelConfig(**state['config']['model']));model.load_state_dict(state['model']);install(model,'native');model.eval()
    initial_hash=weights_sha(model.state_dict());model.requires_grad_(False);model.to(args.device);model.backbone.embeddings.device=args.device
    common=torch.load(args.source/'shared-data-and-initialization.pt',map_location='cpu',weights_only=True)
    train={k:v[:4096] for k,v in common['splits']['train'].items()};dev=common['splits']['validation']
    torch.manual_seed(2026091451);fresh_seg=multiquery_ar(vocab_size=256,num_examples=1024,input_seq_len=64,seed=2026091451,num_kv_pairs=4)
    fresh=dict(inputs=fresh_seg.inputs,labels=fresh_seg.labels);swapped,pairs=value_swap(fresh)
    old_hashes={hashlib.sha256(x.numpy().tobytes()).hexdigest() for d in common['splits'].values() for x in d['inputs']}
    new_hashes=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in fresh['inputs']];assert len(set(new_hashes))==1024 and not set(new_hashes)&old_hashes
    for data in [train,dev,fresh]:
        for x,y in zip(data['inputs'],data['labels']):
            mapping={int(x[i]):int(x[i+1]) for i in range(0,8,2)};assert len(mapping)==4 and len(set(mapping.values()))==4 and int((y!=-100).sum())==4
            for p in torch.where(y!=-100)[0]:assert int(p)>=8 and mapping[int(x[p])]==int(y[p])
    torch.save(dict(train=train,development=dev,fresh=fresh,swapped=swapped,swap_positions=pairs),out/'data.pt')
    save(out/'config.json',dict(backbone_checkpoint_sha256=sha(checkpoint),backbone_hash=initial_hash,backbone_updates=0,
        train_examples=4096,fresh_examples=1024,fresh_seed=2026091451,indexer_seed=2026091460,epochs=40,batch_size=32,
        lr=.001,weight_decay=0,teacher='Full causal softmax on frozen native-route activations',k=8,local=2,
        plan='frozen-router-capacity-plan-2026-09-14.md',fresh_disjoint_from_all_old_splits=True,
        variants=['r16_random','r32_random','r64_random','r32_svd'],zero_update_controls=['exact','r128_copy']))
    save(out/'environment.json',dict(utc=now(),python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device=args.device,
        gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,argv=sys.argv))
    # Numerical CUDA check before any fitting: copied full-rank selector + gathered core.
    torch.manual_seed(2026091460);copied=torch.nn.ModuleList([TokenIndexer(128,128) for _ in range(2)]).to(args.device)
    for idx,layer in zip(copied,model.backbone.layers):idx.initialize(layer.sequence_mixer,'copy')
    x=train['inputs'][:8].to(args.device)
    with torch.no_grad():expected=model(x)
    routing=FrozenRouter(model,copied)
    with torch.no_grad():actual=model(x)
    torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5);routing.restore()
    save(out/'preflight.json',dict(full_rank_forward_pass=True,additional_training_updates=0,max_error=float((actual-expected).abs().max())))
    # Cache teacher features under the unmodified native route, avoiding training-time drift.
    caches=[dict(hidden=[],target=[]) for _ in range(2)]
    def hook_for(i):
        def hook(mha,inputs):
            hidden=inputs[0].detach();caches[i]['hidden'].append(hidden);caches[i]['target'].append(full_teacher(mha,hidden).detach())
        return hook
    hooks=[layer.sequence_mixer.register_forward_pre_hook(hook_for(i)) for i,layer in enumerate(model.backbone.layers)]
    with torch.no_grad():
        for first in range(0,len(train['inputs']),32):cap();model(train['inputs'][first:first+32].to(args.device))
    for hook in hooks:hook.remove()
    caches=[{k:torch.cat(v) for k,v in c.items()} for c in caches]
    save(out/'cache.json',dict(examples=4096,layers=2,bytes=sum(t.numel()*t.element_size() for c in caches for t in c.values()),
        teacher_depends_on_task_labels=False,teacher_backbone_route='native exact top8',teacher_target='full causal QK softmax'))
    controls={'exact':dict(development=evaluate(model,dev,args.device))};completed=[]
    for name,rank,init in [('r16_random',16,'random'),('r32_random',32,'random'),('r64_random',64,'random'),('r32_svd',32,'svd')]:
        folder=out/name;folder.mkdir();set_determinism(2026091460);indexers=torch.nn.ModuleList([TokenIndexer(128,rank) for _ in range(2)]).to(args.device)
        if init=='svd':
            for idx,layer in zip(indexers,model.backbone.layers):idx.initialize(layer.sequence_mixer,init)
        torch.save(indexers.state_dict(),folder/'initial-indexer.pt');optimizer=torch.optim.AdamW(indexers.parameters(),lr=.001,weight_decay=0)
        start=time.perf_counter();started=now();steps=0;curves=[]
        with (folder/'events.jsonl').open('w',encoding='utf-8',buffering=65536) as log:
            def event(kind,**kw):
                log.write(json.dumps(dict(utc=now(),event=kind,elapsed_seconds=time.perf_counter()-start,**kw))+'\n')
                if kind!='optimizer_step':log.flush()
            event('start',method=name,rank=rank,initialization=init)
            try:
                for epoch in range(40):
                    cap();loss_sum=0.
                    for first in range(0,4096,32):
                        optimizer.zero_grad();loss=sum(kl_loss(idx(c['hidden'][first:first+32]),c['target'][first:first+32]) for idx,c in zip(indexers,caches))
                        loss.backward();grad=float(torch.stack([p.grad.square().sum() for p in indexers.parameters()]).sum().sqrt());scalar=float(loss)
                        if not math.isfinite(grad) or not math.isfinite(scalar):raise FloatingPointError('Nonfinite indexer fitting')
                        optimizer.step();steps+=1;loss_sum+=scalar;event('optimizer_step',step=steps,epoch=epoch,kl_sum_layers=scalar,gradient_norm=grad,indexer_input_tokens=steps*32*64)
                    row=dict(epoch=epoch+1,step=steps,mean_kl_sum_layers=loss_sum/128)
                    if (epoch+1)%5==0:
                        routing=FrozenRouter(model,indexers)
                        try:row['development_accuracy']=evaluate(model,dev,args.device)['accuracy']
                        finally:routing.restore()
                        torch.save(dict(indexers=indexers.state_dict(),optimizer=optimizer.state_dict(),rank=rank,init=init,steps=steps,epoch=epoch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None),folder/'checkpoint.pt')
                    curves.append(row);event('epoch',**row);print(json.dumps(dict(method=name,**row)),flush=True)
                r=dict(status='fit_complete',method=name,rank=rank,initialization=init,started_utc=started,finished_utc=now(),
                    optimizer_updates=steps,backbone_updates=0,indexer_input_tokens=steps*32*64,fit_wall_seconds=time.perf_counter()-start,curves=curves)
                save(folder/'fit-result.json',r);event('fit_complete',steps=steps);completed.append(r)
            except Exception:
                event('failed',traceback=traceback.format_exc());torch.save(dict(indexers=indexers.state_dict(),optimizer=optimizer.state_dict(),steps=steps),folder/'failure-checkpoint.pt');raise
        save(out/'progress.json',dict(utc=now(),status='fitting',completed=[r['method'] for r in completed]))
    # All fixed fitting is now over. Evaluate the same fresh examples without further tuning.
    controls['exact'].update(fresh=evaluate(model,fresh,args.device),swapped=evaluate(model,swapped,args.device))
    routing=FrozenRouter(model,copied)
    try:controls['r128_copy']=dict(fresh=evaluate(model,fresh,args.device),swapped=evaluate(model,swapped,args.device))
    finally:routing.restore()
    assert controls['exact']['fresh']['predictions']==controls['r128_copy']['fresh']['predictions']
    results=[]
    for r in completed:
        cap();folder=out/r['method'];weights=torch.load(folder/'checkpoint.pt',map_location=args.device,weights_only=False)
        indexers=torch.nn.ModuleList([TokenIndexer(128,r['rank']) for _ in range(2)]).to(args.device);indexers.load_state_dict(weights['indexers']);routing=FrozenRouter(model,indexers)
        try:
            r.update(fresh=evaluate(model,fresh,args.device),swapped=evaluate(model,swapped,args.device))
            routing.exact_layers={1};r['only_layer0']=evaluate(model,fresh,args.device)
            routing.exact_layers={0};r['only_layer1']=evaluate(model,fresh,args.device)
        finally:routing.restore()
        r['pass_to_joint_diagnostic']=(r['fresh']['accuracy']>=controls['exact']['fresh']['accuracy']-.01 and r['swapped']['accuracy']>=controls['exact']['swapped']['accuracy']-.01)
        r['status']='complete';save(folder/'result.json',r);results.append(r)
    assert weights_sha(model.state_dict())==initial_hash and all(p.grad is None for p in model.parameters())
    result=dict(status='complete',started_utc=read_env_time(out),finished_utc=now(),wall_seconds=time.perf_counter()-timer,backbone_unchanged=True,
        backbone_updates=0,indexer_optimizer_updates=sum(r['optimizer_updates'] for r in results),controls=controls,runs=results,
        scope='Frozen trained 437760-parameter MQAR backbone; token low-rank routing capacity screen. Not from-scratch training, not QSA/MSA reproduction, no efficient-kernel claim.')
    save(out/'result.json',result);save(out/'progress.json',dict(status='complete',utc=now(),completed=[r['method'] for r in results]));save(out/'SUCCESS.json',dict(utc=now(),backbone_unchanged=True))
    save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()])
    print(json.dumps(dict(status='complete',runs=[dict(method=r['method'],fresh=r['fresh']['accuracy'],swapped=r['swapped']['accuracy'],gate=r['pass_to_joint_diagnostic']) for r in results])),flush=True)

def read_env_time(out):return json.loads((out/'environment.json').read_text())['utc']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=p.parse_args()
    try:main(args)
    except Exception:
        if args.output.exists():save(args.output/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()))
        raise
