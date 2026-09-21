"""Frozen 5-arm exact-top-k training screen with complete per-step evidence."""
import argparse,hashlib,json,math,os,random,shutil,signal,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from zoology_entry import ROOT,UPSTREAM,configuration,LanguageModel,prepare_data,multiquery_ar,set_determinism
from zoology_sparse_schedule import SCHEDULES,install,set_epoch,retained_edges

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def tensor_hash(state):
    h=hashlib.sha256()
    for k,t in sorted(state.items()):h.update(k.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def evaluate(model,x,y,batch=32):
    model.eval();correct=0;count=0;loss=0.;predictions=[]
    for start in range(0,len(x),batch):
        target=y[start:start+batch];logits=model(x[start:start+batch]);mask=target!=-100
        p=logits.argmax(-1);correct+=int((p[mask]==target[mask]).sum());count+=int(mask.sum())
        loss+=float(torch.nn.functional.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='sum'))
        predictions.extend(p[mask].cpu().tolist())
    return dict(accuracy=correct/count,nll=loss/count,answers=count,predictions=predictions)

def prepare(output,length):
    cfg=configuration(length);set_determinism(123);model=LanguageModel(cfg.model)
    init={k:v.clone() for k,v in model.state_dict().items()};train,val=prepare_data(cfg.data)
    splits={}
    for name,dl in [('train',train),('validation',val)]:
        data=dl.dataset.segments[0];splits[name]=dict(inputs=data.inputs,labels=data.labels)
    # This seed and split are frozen, stored now but only evaluated after training.
    torch.manual_seed(2026091440)
    fresh=multiquery_ar(vocab_size=256,num_examples=1000,input_seq_len=length,seed=2026091440,num_kv_pairs=4)
    splits['fresh']=dict(inputs=fresh.inputs,labels=fresh.labels)
    hashes={}
    for name,data in splits.items():
        rows=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in data['inputs']]
        assert len(rows)==len(set(rows));hashes[name]=set(rows)
        for x,y in zip(data['inputs'],data['labels']):
            mapping={int(x[i]):int(x[i+1]) for i in range(0,8,2)}
            assert len(mapping)==4 and len(set(mapping.values()))==4 and int((y!=-100).sum())==4
            for p in torch.where(y!=-100)[0]:assert int(p)>=8 and mapping[int(x[p])]==int(y[p])
    assert not hashes['train']&hashes['validation'] and not hashes['fresh']&(hashes['train']|hashes['validation'])
    torch.save(dict(initial_state=init,splits=splits),output/'shared-data-and-initialization.pt')
    save(output/'data-audit.json',dict(input_rows_disjoint=True,labels_checked=True,initial_hash=tensor_hash(init),
        seed=123,fresh_seed=2026091440,length=length,splits={k:len(v['inputs']) for k,v in splits.items()}))
    return cfg,init,splits

def run_one(folder,method,cfg,init,splits,args):
    folder.mkdir(exist_ok=False);started=now();timer=time.perf_counter()
    event_file=(folder/'events.jsonl').open('a',encoding='utf-8',buffering=65536)
    def event(kind,**fields):
        event_file.write(json.dumps(dict(utc=now(),elapsed_seconds=time.perf_counter()-timer,event=kind,**fields))+'\n')
        if kind!='optimizer_step':event_file.flush()
    set_determinism(123);model=LanguageModel(cfg.model.model_copy(deep=True));model.load_state_dict(init);install(model,method)
    assert tensor_hash(model.state_dict())==tensor_hash(init)
    model.to(args.device);model.backbone.embeddings.device=args.device
    data={k:{n:t.to(args.device) for n,t in v.items()} for k,v in splits.items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.)
    if args.device=='cuda':torch.cuda.manual_seed_all(123);torch.cuda.reset_peak_memory_stats()
    steps=0;examples=0;curves=[];logical=0;train_seconds=0.;failure=None
    save(folder/'config.json',dict(method=method,epochs=args.epochs,seed=123,k=8,local=2,base=cfg.model_dump(serialize_as_any=True),
        device=args.device,scope='Exact dense QK scoring and dense masked AV: mechanism reference only, no sparse kernel speed claim'))
    event('start',method=method,initial_hash=tensor_hash(init))
    try:
        for epoch in range(args.epochs):
            if time.perf_counter()-timer>args.max_seconds:raise TimeoutError('Per-run wall-clock cap reached')
            set_epoch(model,epoch);model.train();train_loss=0.;train_answers=0
            for start in range(0,len(data['train']['inputs']),32):
                ts=time.perf_counter();x=data['train']['inputs'][start:start+32];y=data['train']['labels'][start:start+32]
                optimizer.zero_grad();logits=model(x);loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten())
                loss.backward();grad=float(torch.stack([p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt())
                scalar=float(loss)
                if not math.isfinite(grad) or not math.isfinite(scalar):raise FloatingPointError('Nonfinite training')
                optimizer.step();steps+=1;examples+=len(x)
                if args.device=='cuda':torch.cuda.synchronize()
                duration=time.perf_counter()-ts;train_seconds+=duration
                count=int((y!=-100).sum());train_loss+=scalar*count;train_answers+=count
                layer_edges=sum(retained_edges(args.length,args.length if epoch<SCHEDULES[method][i] else 8) for i in range(2))
                logical+=len(x)*layer_edges
                event('optimizer_step',step=steps,epoch=epoch,loss=scalar,gradient_norm=grad,learning_rate=optimizer.param_groups[0]['lr'],
                    batch_examples=len(x),input_tokens=examples*args.length,
                    cumulative_retained_edges=logical,step_seconds=duration)
            validation=evaluate(model,**{'x':data['validation']['inputs'],'y':data['validation']['labels']})
            row=dict(epoch=epoch+1,step=steps,train_nll=train_loss/train_answers,validation_accuracy=validation['accuracy'],
                validation_nll=validation['nll'],logical_retained_edges=logical)
            curves.append(row);event('validation',**row);print(json.dumps(dict(method=method,**row)),flush=True)
            scheduler.step()
            checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,steps=steps,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None,
                numpy_rng=np.random.get_state(),python_rng=random.getstate(),config=cfg.model_dump(serialize_as_any=True),method=method)
            torch.save(checkpoint,folder/'checkpoint.pt')
        set_epoch(model,args.epochs);before=tensor_hash(model.state_dict());fresh=evaluate(model,data['fresh']['inputs'],data['fresh']['labels'])
        assert before==tensor_hash(model.state_dict())
        result=dict(status='complete',method=method,started_utc=started,finished_utc=now(),epochs=args.epochs,updates=steps,
            input_tokens=args.epochs*10000*args.length,supervised_answers=args.epochs*40000,training_seconds=train_seconds,
            wall_seconds=time.perf_counter()-timer,initial_hash=tensor_hash(init),final_hash=before,curves=curves,fresh=fresh,
            cumulative_retained_edges=logical,peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
            scope='One seed, fixed epochs, exact-score dense-mask reference; not a QSA indexer, not proof of sparse training speed or novelty')
        save(folder/'result.json',result);event('complete',fresh_accuracy=fresh['accuracy'])
    except Exception:
        failure=traceback.format_exc();event('failed',traceback=failure)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),steps=steps),folder/'failure-checkpoint.pt')
        save(folder/'result.json',dict(status='failed',method=method,started_utc=started,finished_utc=now(),updates=steps,traceback=failure))
        raise
    finally:
        event_file.close()
        save(folder/'manifest.json',[dict(path=p.name,sha256=sha(p)) for p in sorted(folder.iterdir()) if p.is_file()])
    return result

def main(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    def stopped(signum,frame):raise RuntimeError('Experiment interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,stopped)
    signal.signal(signal.SIGINT,stopped)
    torch.set_num_threads(4)
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(.50)
    snapshot=output/'source';snapshot.mkdir()
    for name in ['run_sparse_schedule.py','zoology_sparse_schedule.py','zoology_entry.py']:shutil.copy2(ROOT/'src'/name,snapshot/name)
    shutil.copytree(UPSTREAM,snapshot/'upstream',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    save(output/'environment.json',dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device=args.device,
        gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,argv=sys.argv,started_utc=now(),wandb_mode=os.environ.get('WANDB_MODE')))
    cfg,init,data=prepare(output,args.length);results=[]
    for method in args.methods:
        result=run_one(output/method,method,cfg,init,data,args);results.append({k:v for k,v in result.items() if k not in ['curves','fresh'] }|{'fresh_accuracy':result['fresh']['accuracy']})
        save(output/'progress.json',dict(status='running',completed=results,utc=now()))
        if args.device=='cuda':torch.cuda.empty_cache()
    save(output/'result.json',dict(status='complete',runs=results,finished_utc=now()))
    save(output/'progress.json',dict(status='complete',completed=results,utc=now()))
    save(output/'manifest.json',[dict(path=p.relative_to(output).as_posix(),sha256=sha(p)) for p in sorted(output.rglob('*')) if p.is_file()])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--length',type=int,default=64);p.add_argument('--epochs',type=int,default=40);p.add_argument('--max-seconds',type=int,default=3600)
    p.add_argument('--methods',nargs='+',choices=list(SCHEDULES),default=list(SCHEDULES));main(p.parse_args())
