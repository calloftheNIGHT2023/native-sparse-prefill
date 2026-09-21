"""Instrument the pinned upstream Trainer without replacing its optimization loop."""
import argparse,hashlib,json,math,os,random,shutil,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from zoology_entry import ROOT,UPSTREAM,configuration,LanguageModel,prepare_data,multiquery_ar,Trainer,set_determinism

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def tensor_hash(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

class LocalLogger:
    def __init__(self,out):self.out=out;self.trainer=None
    def log(self,metrics):
        t=self.trainer
        if 'train/loss' in metrics:
            t.event('optimizer_step',step=t.steps,epoch=metrics['epoch'],loss=float(metrics['train/loss']),
                grad_norm=t.last_grad_norm,learning_rate=t.optimizer.param_groups[0]['lr'],
                examples=t.examples,supervised_answers=t.answers,input_tokens=t.input_tokens,
                step_seconds=t.last_step_seconds,batch_examples=t.last_batch_size)
        elif 'valid/accuracy' in metrics:
            t.curves.append(dict(step=t.steps,**metrics));t.event('validation',step=t.steps,**metrics)
            print(json.dumps(dict(epoch=metrics['epoch'],step=t.steps,validation_accuracy=metrics['valid/accuracy'],validation_loss=metrics['valid/loss'])),flush=True)

class LoggedTrainer(Trainer):
    def __init__(self,*args,out,cfg,max_seconds,**kwargs):
        super().__init__(*args,**kwargs);self.out=out;self.cfg=cfg;self.max_seconds=max_seconds
        self.timer=time.perf_counter();self.started=utc();self.steps=0;self.examples=0;self.answers=0;self.input_tokens=0
        self.training_seconds=0.;self.curves=[];self.hook_handles=[];self.last_epoch=-1
        self.logger.trainer=self
    def event(self,kind,**kwargs):
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=utc(),elapsed_seconds=time.perf_counter()-self.timer,event=kind,**kwargs))+'\n')
    def compute_loss(self,inputs,targets):
        if self.model.training:
            self.step_started=time.perf_counter();self.last_batch_size=len(inputs);self.last_tokens=inputs.numel();self.last_answers=int((targets!=-100).sum())
        result=super().compute_loss(inputs,targets)
        if not torch.isfinite(result[0]):raise FloatingPointError('Non-finite upstream loss')
        return result
    def train_epoch(self,epoch_idx):
        self.last_epoch=epoch_idx
        if not self.hook_handles:
            def before(opt,args,kwargs):
                norms=[p.grad.detach().square().sum() for p in self.model.parameters() if p.grad is not None]
                self.last_grad_norm=float(torch.stack(norms).sum().sqrt())
                if not math.isfinite(self.last_grad_norm):raise FloatingPointError('Non-finite gradient')
            def after(opt,args,kwargs):
                self.steps+=1;self.examples+=self.last_batch_size;self.answers+=self.last_answers;self.input_tokens+=self.last_tokens
                self.last_step_seconds=time.perf_counter()-self.step_started;self.training_seconds+=self.last_step_seconds
            self.hook_handles=[self.optimizer.register_step_pre_hook(before),self.optimizer.register_step_post_hook(after)]
        if time.perf_counter()-self.timer>self.max_seconds:raise TimeoutError('CPU stage time cap at epoch boundary')
        return super().train_epoch(epoch_idx)
    def checkpoint(self,status):
        torch.save(dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),
            epoch=self.last_epoch,steps=self.steps,status=status,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            config=self.cfg,scope='Checkpoint saved after epoch validation before upstream scheduler step; not a ready-made resume CLI'),self.out/'checkpoint.pt')
    def test(self,epoch_idx):
        result=super().test(epoch_idx);self.checkpoint('epoch_validated');return result

@torch.no_grad()
def independent_evaluation(model,out,input_seq_len=64):
    seed=2026091425;torch.manual_seed(seed)
    fresh=multiquery_ar(vocab_size=256,num_examples=1000,input_seq_len=input_seq_len,seed=seed,num_kv_pairs=4)
    torch.save(dict(inputs=fresh.inputs,labels=fresh.labels),out/'fresh-evaluation-data.pt')
    model.eval();preds=[];correct=0;count=0;loss_sum=0.;seq=[]
    for start in range(0,1000,32):
        x=fresh.inputs[start:start+32];y=fresh.labels[start:start+32];lg=model(x);p=lg.argmax(-1);mask=y!=-100
        correct+=int((p[mask]==y[mask]).sum());count+=int(mask.sum());loss_sum+=float(torch.nn.functional.cross_entropy(lg.flatten(0,1),y.flatten(),reduction='sum'))
        preds.extend(p[mask].tolist());seq.extend(((p==y)&mask).sum(-1).div(mask.sum(-1)).tolist())
    return dict(seed=seed,examples=1000,supervised_answers=count,accuracy=correct/count,nll=loss_sum/count,sequence_scores=seq,predictions=preds)

def main(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);cfg=configuration(args.sequence_length);torch.set_num_threads(4);set_determinism(cfg.seed)
    source=out/'source-snapshot';shutil.copytree(UPSTREAM,source/'upstream')
    for name in ['run_zoology_baseline.py','zoology_entry.py']:shutil.copy2(ROOT/'src'/name,source/name)
    # Match upstream train(config): initialize the model BEFORE generating random filler data.
    model=LanguageModel(cfg.model);initial=tensor_hash(model.state_dict());train_dl,val_dl=prepare_data(cfg.data)
    data_info={}
    hashes={}
    for split,dl in [('train',train_dl),('validation',val_dl)]:
        segment=dl.dataset.segments[0];torch.save(dict(inputs=segment.inputs,labels=segment.labels),out/(split+'-data.pt'))
        rowhash=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in segment.inputs];hashes[split]=set(rowhash)
        data_info[split]=dict(examples=len(segment),batches=len(dl),supervised_answers=int((segment.labels!=-100).sum()),sha256=sha(out/(split+'-data.pt')))
        assert len(rowhash)==len(hashes[split])
        for x,y in zip(segment.inputs,segment.labels):
            mapping={int(x[i]):int(x[i+1]) for i in range(0,8,2)}
            assert len(mapping)==4 and int((y!=-100).sum())==4
            for q in torch.where(y!=-100)[0]:assert int(q)>=8 and mapping[int(x[q])]==int(y[q])
    assert not hashes['train']&hashes['validation']
    frozen=cfg.model_dump(serialize_as_any=True);save(out/'config.json',frozen);save(out/'data-audit.json',dict(splits=data_info,input_rows_disjoint=True,labels_checked_from_prefix=True))
    save(out/'environment.json',dict(torch=str(torch.__version__),numpy=np.__version__,device='cpu',threads=4,parameters=sum(p.numel() for p in model.parameters()),
        upstream_commit='1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb',wandb_mode=os.environ['WANDB_MODE']))
    logger=LocalLogger(out)
    task=LoggedTrainer(model=model,train_dataloader=train_dl,test_dataloader=val_dl,input_type=cfg.input_type,max_epochs=cfg.max_epochs,
        learning_rate=cfg.learning_rate,weight_decay=cfg.weight_decay,early_stopping_metric=cfg.early_stopping_metric,
        early_stopping_threshold=cfg.early_stopping_threshold,slice_keys=cfg.slice_keys,loss_type=cfg.loss_type,device='cpu',logger=logger,
        out=out,cfg=frozen,max_seconds=args.max_seconds)
    task.event('start',initial_hash=initial,parameters=sum(p.numel() for p in model.parameters()))
    try:
        task.fit();task.checkpoint('complete');before=tensor_hash(model.state_dict());fresh=independent_evaluation(model,out,args.sequence_length)
        assert before==tensor_hash(model.state_dict())
        saved=torch.load(out/'fresh-evaluation-data.pt',weights_only=True)
        freshhash={hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in saved['inputs']}
        assert len(freshhash)==1000 and not freshhash&(hashes['train']|hashes['validation'])
        for x,y in zip(saved['inputs'],saved['labels']):
            mapping={int(x[i]):int(x[i+1]) for i in range(0,8,2)}
            assert len(mapping)==4 and len(set(mapping.values()))==4 and int((y!=-100).sum())==4
            for q in torch.where(y!=-100)[0]:assert int(q)>=8 and mapping[int(x[q])]==int(y[q])
        result=dict(status='complete',started_utc=task.started,finished_utc=utc(),epochs=task.last_epoch+1,updates=task.steps,
            examples_seen=task.examples,input_tokens=task.input_tokens,supervised_answers=task.answers,training_seconds=task.training_seconds,
            initial_hash=initial,final_hash=before,curves=task.curves,fresh_evaluation=fresh,
            upstream_stop_threshold_reached=task.curves[-1]['valid/accuracy']>.99,
            independent_fresh_at_least_99=fresh['accuracy']>=.99,
            parameters=sum(p.numel() for p in model.parameters()),gpu_jobs_started=0,sequence_length=args.sequence_length,
            fresh_labels_checked=True,fresh_input_rows_disjoint=True,
            scope='Pinned public basic configuration on Windows CPU, with declared sequence length and matching absolute-position table. Independent fresh set was not used for stopping; not full-paper reproduction, length extrapolation, or a new method.')
        save(out/'result.json',result);task.event('complete',fresh_accuracy=fresh['accuracy'])
        save(out/'manifest.json',[dict(path=str(p.relative_to(out)),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()])
        print(json.dumps({k:v for k,v in result.items() if k not in ['curves','fresh_evaluation']}),flush=True)
        print(json.dumps(dict(fresh_accuracy=fresh['accuracy'])),flush=True)
    except Exception:
        task.event('failed',traceback=traceback.format_exc());task.checkpoint('failed');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=int,default=3600);p.add_argument('--sequence-length',type=int,default=64);main(p.parse_args())
