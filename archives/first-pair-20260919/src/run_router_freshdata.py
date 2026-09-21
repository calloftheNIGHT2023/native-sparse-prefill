"""Exploratory dense KV16 data-refresh control after a failed fixed-data control."""
import argparse,hashlib,json,math,random,shutil,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from zoology_entry import ROOT,UPSTREAM,multiquery_ar,set_determinism
from run_frozen_router import now,save,sha,weights_sha,evaluate
from router_pressure_models import build,load_checkpoint
from run_router_pressure import atomic_checkpoint

def generated(seed,rows):
    tr=torch.get_rng_state();nr=np.random.get_state()
    try:
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        d=multiquery_ar(vocab_size=256,num_examples=rows,input_seq_len=128,num_kv_pairs=16,seed=seed)
        return dict(inputs=d.inputs,labels=d.labels)
    finally:torch.set_rng_state(tr);np.random.set_state(nr)

def row_hashes(data):return [hashlib.sha256(x.numpy().tobytes()).digest() for x in data['inputs']]
def tensor_hash(data):
    h=hashlib.sha256()
    for key in ['inputs','labels']:h.update(key.encode());h.update(data[key].numpy().tobytes())
    return h.hexdigest()

def validate(data):
    x,y=data['inputs'],data['labels'];assert x.shape==y.shape and x.shape[1]==128
    assert torch.all((y!=-100).sum(-1)==16)
    bank=x[:,:32:2];values=x[:,1:32:2]
    assert torch.all(bank.sort(-1).values[:,1:]!=bank.sort(-1).values[:,:-1]);assert torch.all(values.sort(-1).values[:,1:]!=values.sort(-1).values[:,:-1])
    b,q=torch.where(y!=-100);matches=bank[b]==x[b,q,None];assert torch.all(matches.sum(-1)==1) and torch.all(q>=32)
    assert torch.equal(values[b,matches.to(torch.int32).argmax(-1)],y[b,q])

def swapped(data):
    d=dict(inputs=data['inputs'].clone(),labels=torch.full_like(data['labels'],-100));positions=[]
    for i,(x,y) in enumerate(zip(data['inputs'],data['labels'])):
        q=int(torch.where(y!=-100)[0][-1]);source=int(torch.where(x[:32:2]==x[q])[0][0])*2+1;alt=(source+2)%32
        d['inputs'][i,source]=x[alt];d['inputs'][i,alt]=x[source];d['labels'][i,q]=x[alt];positions.append([q,source,alt])
        assert int((d['inputs'][i]!=x).sum())==2 and torch.equal(d['inputs'][i,32:],x[32:])
    return d,positions

def main(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.output;out.mkdir(exist_ok=False);timer=time.perf_counter();started=now();snap=out/'source';snap.mkdir()
    for name in ['run_router_freshdata.py','run_router_pressure.py','router_pressure_models.py','router_recipe_baselines.py','run_joint_token_router.py','run_frozen_router.py','frozen_routing.py','joint_token_routing.py','zoology_entry.py','zoology_sparse_schedule.py']:shutil.copy2(ROOT/'src'/name,snap/name)
    shutil.copytree(UPSTREAM,snap/'upstream',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));shutil.copy2(ROOT/'docs/router-freshdata-plan-2026-09-14.md',out/'plan.md')
    old=ROOT/'results/router-pressure-v0';init=torch.load(old/'initialization.pt',map_location='cpu',weights_only=True);base=json.loads((ROOT/'data/router-pressure-kv16-v0/config.json').read_text());previous=torch.load(ROOT/'data/router-pressure-kv16-v0/data.pt',map_location='cpu',weights_only=True)
    fresh=generated(2026091591,1024);validate(fresh);swap,positions=swapped(fresh);torch.save(dict(test=fresh,swapped=swap,swap_positions=positions),out/'new-test.pt');shutil.copy2(old/'initialization.pt',out/'initialization.pt')
    seen=set();reserved=set()
    for key in ['train','development','test','swapped']:reserved.update(row_hashes(previous[key]))
    hashes=row_hashes(fresh);assert len(set(hashes))==1024 and not set(hashes)&reserved;reserved.update(hashes);reserved.update(row_hashes(swap))
    old_ck=old/'dense/checkpoint.pt';save(out/'environment.json',dict(utc=started,python=sys.version,torch=str(torch.__version__),numpy=np.__version__,gpu=torch.cuda.get_device_name(),plan_sha256=sha(out/'plan.md'),old_result_sha256=sha(old/'result.json'),old_checkpoint_sha256=sha(old_ck),old_initialization_sha256=sha(old/'initialization.pt'),old_data_sha256=sha(ROOT/'data/router-pressure-kv16-v0/data.pt')))
    m,ix,route=build(base,init['model'],init['indexers'],'dense','cuda');assert weights_sha(m.state_dict())==weights_sha(init['model']);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.1);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=0.);set_determinism(123)
    # Reuse the previously verified dense wrapper, checking this initialization on both devices.
    reference,_,rr=build(base,init['model'],init['indexers'],'dense','cpu');reference.eval();m.eval()
    with torch.no_grad():torch.testing.assert_close(reference(fresh['inputs'][:2]),m(fresh['inputs'][:2].cuda()).cpu(),rtol=3e-4,atol=3e-5)
    rr.restore();save(out/'preflight.json',dict(dense_cpu_cuda_forward_match=True,optimizer_updates=0));set_determinism(123)
    data_dir=out/'training-data';data_dir.mkdir();steps=0;epoch=-1;curves=[];total_step_seconds=0.;data_seconds=0.;generated_seeds=[]
    def checkpoint(boundary):return dict(method='dense',data_recipe='fresh_per_epoch',config=base,model=m.state_dict(),indexers=ix.state_dict(),main_optimizer=opt.state_dict(),indexer_optimizer=None,scheduler=sched.state_dict(),epoch=epoch,steps=steps,epoch_boundary=boundary,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    with (out/'events.jsonl').open('w',encoding='utf-8',buffering=65536) as log:
        def event(kind,**data):
            log.write(json.dumps(dict(utc=now(),event=kind,elapsed_seconds=time.perf_counter()-timer,**data))+'\n')
            if kind!='optimizer_step':log.flush()
        event('start',epochs=80,initial_backbone_sha=weights_sha(m.state_dict()))
        try:
            for epoch in range(80):
                tick=time.perf_counter();seed=2026091501+epoch;train=generated(seed,10000);validate(train);hashes=row_hashes(train);unique=set(hashes)
                assert len(unique)==10000 and not unique&reserved and not unique&seen;seen.update(unique)
                np.savez_compressed(data_dir/f'epoch{epoch+1:03d}.npz',inputs=train['inputs'].numpy(),labels=train['labels'].numpy())
                meta=dict(epoch=epoch+1,seed=seed,tensor_sha256=tensor_hash(train),rows=10000,new_unique_rows=True);generated_seeds.append(meta);event('data_generated',**meta)
                train={k:v.cuda() for k,v in train.items()};dt=time.perf_counter()-tick;data_seconds+=dt
                m.train();route.collect_aux=False;ce_sum=0.;epoch_step_seconds=0.
                for first in range(0,10000,32):
                    if time.perf_counter()-timer>1200:raise TimeoutError('Fresh-data control 20-minute wall cap')
                    tick=time.perf_counter();opt.zero_grad();route.reset();x=train['inputs'][first:first+32];y=train['labels'][first:first+32];logits=m(x);ce=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten());ce.backward();grad=float(torch.stack([p.grad.square().sum() for p in m.parameters() if p.grad is not None]).sum().sqrt());loss=float(ce.detach());assert math.isfinite(loss) and math.isfinite(grad);opt.step();torch.cuda.synchronize();sec=time.perf_counter()-tick;total_step_seconds+=sec;epoch_step_seconds+=sec;steps+=1;ce_sum+=loss*len(x)
                    event('optimizer_step',global_step=steps,epoch=epoch+1,batch_examples=len(x),input_tokens=x.numel(),supervised_answers=int((y!=-100).sum()),ce=loss,main_gradient_norm=grad,main_lr=opt.param_groups[0]['lr'],step_wall_seconds=sec)
                m.eval();tick=time.perf_counter();dev=evaluate(m,previous['development'],'cuda');dev_seconds=time.perf_counter()-tick;sched.step();tick=time.perf_counter();atomic_checkpoint(out/'checkpoint.pt',checkpoint(True))
                if epoch+1 in (20,40,80):shutil.copy2(out/'checkpoint.pt',out/f'checkpoint_epoch{epoch+1:03d}.pt')
                row=dict(epoch=epoch+1,global_step=steps,train_nll=ce_sum/10000,development_accuracy=dev['accuracy'],development_nll=dev['nll'],train_step_seconds=epoch_step_seconds,data_seconds=dt,development_seconds=dev_seconds,checkpoint_seconds=time.perf_counter()-tick);curves.append(row);event('epoch',**row);save(out/'progress.json',dict(utc=now(),status='running',**row));print(json.dumps(row),flush=True)
            fit_finished=now();event('fit_complete',global_step=steps);save(out/'data-seeds.json',generated_seeds)
        except Exception:
            atomic_checkpoint(out/'failure-checkpoint.pt',checkpoint(False));event('failed',global_step=steps);save(out/'FAILURE.json',dict(utc=now(),steps=steps,traceback=traceback.format_exc()));raise
    # All optimizer updates have finished; old final weights are an independently evaluated reference.
    evaluations=[]
    for label,path in [('fixed_data_epoch80',old_ck),('fresh_data_epoch80',out/'checkpoint.pt')]:
        c=torch.load(path,map_location='cpu',weights_only=False);em,ei,er=load_checkpoint(c,'cuda');row=dict(condition=label,checkpoint_sha256=sha(path))
        for key,dataset in [('test',fresh),('swapped',swap)]:row[key]=evaluate(em,dataset,'cuda')
        er.restore();evaluations.append(row)
    gate=curves[-1]['development_accuracy']>=.99 and all(evaluations[1][k]['accuracy']>=.99 for k in ['test','swapped'])
    result=dict(status='complete',started_utc=started,fit_finished_utc=fit_finished,finished_utc=now(),main_updates=steps,indexer_updates=0,input_tokens=80*1280000,supervised_answers=80*160000,unique_training_sequences=len(seen),wall_seconds=time.perf_counter()-timer,train_step_seconds=total_step_seconds,data_generation_and_save_seconds=data_seconds,curves=curves,evaluations=evaluations,gate=gate,final_backbone_sha=weights_sha(m.state_dict()),scope='Exploratory data-refresh control after fixed-data baseline failure; same exposures, 80x unique examples; no novel sparse method.')
    route.restore();save(out/'result.json',result);save(out/'SUCCESS.json',dict(utc=now()));save(out/'progress.json',dict(utc=now(),status='complete',gate=gate));save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(status='complete',gate=gate,evaluations=[dict(condition=e['condition'],test=e['test']['accuracy'],swapped=e['swapped']['accuracy']) for e in evaluations])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    def interrupted(sig,frame):raise RuntimeError('Signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:main(args)
    except Exception:
        if args.output.exists():save(args.output/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()))
        raise
