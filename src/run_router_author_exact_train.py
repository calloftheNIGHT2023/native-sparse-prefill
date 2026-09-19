"""Exact-QK top8 from initialization: a missing control, not an efficient sparse method."""
import argparse,json,math,random,shutil,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from zoology_entry import ROOT,UPSTREAM,set_determinism
from router_author_control import author_configs,make_model,generated,validate,row_hashes,interventions,selected_logits,evaluate
from router_author_sparse import build,METHODS
from zoology_sparse_schedule import install,retained_edges
from run_frozen_router import now,save,sha,weights_sha
from run_router_author_control import checkpoint_save

def exact_model(cfg,initial,device):
    m=make_model(cfg,initial,device);install(m,'native');return m

def preflight(cfg,initial):
    d=generated(2026091641,2);a=exact_model(cfg,initial,'cpu');b=exact_model(cfg,initial,'cuda');a.eval();b.eval();x=d['inputs'];y=d['labels'];oa,ta=selected_logits(a,x,y);ob,tb=selected_logits(b,x.cuda(),y.cuda());torch.testing.assert_close(oa,ob.cpu(),rtol=3e-4,atol=3e-5);error=float((oa-ob.cpu()).detach().abs().max());b.train();loss=torch.nn.functional.cross_entropy(*selected_logits(b,x.cuda(),y.cuda()));loss.backward();assert all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None)
    return dict(utc=now(),maximum_cpu_cuda_logit_error=error,finite_gradients=True,optimizer_updates=0)

def main(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;old=ROOT/'results/router-author-control-v0';prior=ROOT/'results/router-author-sparse-v0';assert (prior/'SUCCESS.json').exists() and not (prior/'FAILURE.json').exists();out=args.output;out.mkdir(exist_ok=False);timer=time.perf_counter();started=now();cfg=author_configs()[1];initial=torch.load(old/'initialization.pt',map_location='cpu',weights_only=True);data=torch.load(old/'data.pt',map_location='cpu',weights_only=True);shutil.copy2(old/'initialization.pt',out/'initialization.pt');shutil.copy2(ROOT/'docs/router-author-exact-train-plan-2026-09-14.md',out/'plan.md');snap=out/'source';snap.mkdir()
    for name in ['run_router_author_exact_train.py','router_author_control.py','router_author_sparse.py','run_router_author_control.py','router_recipe_baselines.py','router_pressure_models.py','run_joint_token_router.py','run_frozen_router.py','joint_token_routing.py','frozen_routing.py','zoology_entry.py','zoology_sparse_schedule.py']:shutil.copy2(ROOT/'src'/name,snap/name)
    shutil.copytree(UPSTREAM,snap/'upstream',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));save(out/'config.json',cfg);save(out/'preflight.json',preflight(cfg,initial));save(out/'environment.json',dict(utc=started,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),python=sys.version,tf32=False));save(out/'provenance.json',dict(utc=started,training_data_sha256=sha(old/'data.pt'),initial_backbone_sha=weights_sha(initial),plan_sha256=sha(out/'plan.md'),previous_batch_result_sha256=sha(prior/'result.json'),test_seed=2026091642,noise_seed=2026091643,technical_optimizer_updates=0))
    m=exact_model(cfg,initial,'cuda');assert weights_sha(m.state_dict())==weights_sha(initial);opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.1);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=64);set_determinism(123);train={k:v.cuda() for k,v in data['train'].items()};steps=0;epoch=-1;curves=[];seconds_total=0.;fit_timer=time.perf_counter();fit_start=now()
    def ck(boundary):return dict(config=cfg,model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,steps=steps,epoch_boundary=boundary,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    with (out/'events.jsonl').open('w',encoding='utf-8',buffering=65536) as f:
        def event(kind,**kw):
            f.write(json.dumps(dict(utc=now(),event=kind,elapsed_seconds=time.perf_counter()-fit_timer,**kw))+'\n')
            if kind!='optimizer_step':f.flush()
        event('start',method='exact_native')
        try:
            for epoch in range(19):
                m.train();ce_sum=0.;correct=0;epoch_seconds=0.
                for first in range(0,100000,256):
                    if time.perf_counter()-timer>900:raise TimeoutError('Exact-native fixed wall cap')
                    tick=time.perf_counter();opt.zero_grad();x=train['inputs'][first:first+256];y=train['labels'][first:first+256];logits,targets=selected_logits(m,x,y);ce=torch.nn.functional.cross_entropy(logits,targets);ce.backward();grad=float(torch.stack([p.grad.square().sum() for p in m.parameters() if p.grad is not None]).sum().sqrt());cv=float(ce.detach());assert math.isfinite(cv) and math.isfinite(grad);opt.step();torch.cuda.synchronize();seconds=time.perf_counter()-tick;steps+=1;epoch_seconds+=seconds;seconds_total+=seconds;ce_sum+=cv*len(targets);correct+=int((logits.detach().argmax(-1)==targets).sum());event('optimizer_step',global_step=steps,epoch=epoch+1,batch_examples=len(x),input_tokens=x.numel(),supervised_answers=len(targets),ce=cv,main_gradient_norm=grad,main_lr=opt.param_groups[0]['lr'],step_wall_seconds=seconds,full_main_score_entries=2*len(x)*256*256,selected_attention_edges=2*len(x)*retained_edges(256,8),indexer_updates=0)
                tick=time.perf_counter();dev=evaluate(m,data['development'],'cuda');dev_seconds=time.perf_counter()-tick;sched.step();tick=time.perf_counter();checkpoint_save(out/'checkpoint.pt',ck(True))
                if epoch+1 in [4,8,16,19]:shutil.copy2(out/'checkpoint.pt',out/f'checkpoint_epoch{epoch+1:03d}.pt')
                row=dict(epoch=epoch+1,global_step=steps,train_nll=ce_sum/1600000,train_accuracy=correct/1600000,development_accuracy=dev['accuracy'],development_nll=dev['nll'],train_step_seconds=epoch_seconds,development_seconds=dev_seconds,checkpoint_seconds=time.perf_counter()-tick);curves.append(row);event('epoch',**row);save(out/'progress.json',dict(utc=now(),status='fitting',**row));print(json.dumps(row),flush=True)
            fit=dict(method='exact_native',started_utc=fit_start,finished_utc=now(),epochs=19,main_updates=steps,indexer_updates=0,input_tokens=486400000,supervised_answers=30400000,train_step_seconds=seconds_total,wall_seconds=time.perf_counter()-fit_timer,curves=curves,final_backbone_sha=weights_sha(m.state_dict()),development_gate=curves[-1]['development_accuracy']>=.99);save(out/'fit-result.json',fit);event('fit_complete',global_step=steps)
        except Exception:
            checkpoint_save(out/'failure-checkpoint.pt',ck(False));save(out/'FAILURE.json',dict(utc=now(),steps=steps,traceback=traceback.format_exc()));event('failed',global_step=steps);raise
    fresh_test=generated(2026091642,1024);validate(fresh_test);hh=row_hashes(fresh_test);assert len(hh)==1024
    for key in ['train','development','test']:assert not hh&row_hashes(data[key])
    prev_data=torch.load(prior/'evaluation-data.pt',map_location='cpu',weights_only=True);assert not hh&row_hashes(prev_data['test']);swapped,noisy,pairs=interventions(fresh_test);mask=fresh_test['inputs']==0;fill=torch.randint(0,8192,fresh_test['inputs'].shape,generator=torch.Generator().manual_seed(2026091643));noisy['inputs']=fresh_test['inputs'].clone();noisy['inputs'][mask]=fill[mask];validate(noisy,False);fresh=dict(test=fresh_test,swapped=swapped,noisy=noisy,swap_positions=pairs);torch.save(fresh,out/'evaluation-data.pt');save(out/'data-audit.json',dict(test_rows=1024,new_rows=True,test_seed=2026091642,noise_seed=2026091643,evaluation_data_sha256=sha(out/'evaluation-data.pt')))
    rows=[];references=[]
    for method in ['dense_existing',*METHODS,'exact_native']:
        route=None
        if method=='exact_native':model=m;dev_accuracy=curves[-1]['development_accuracy'];path=out/'checkpoint.pt';source_fit=fit
        elif method=='dense_existing':
            path=old/'lr2/checkpoint.pt';c=torch.load(path,map_location='cpu',weights_only=False);model=make_model(cfg,c['model'],'cuda');source_fit=json.loads((old/'lr2/fit-result.json').read_text());dev_accuracy=source_fit['curves'][-1]['development_accuracy']
        else:
            path=prior/method/'checkpoint.pt';c=torch.load(path,map_location='cpu',weights_only=False);model,ix,route=build(cfg,c['model'],c['indexers'],method,'cuda');route.epoch=19;route.collect_aux=False;source_fit=json.loads((prior/method/'fit-result.json').read_text());dev_accuracy=source_fit['curves'][-1]['development_accuracy']
        before=weights_sha(model.state_dict());assert before==source_fit['final_backbone_sha'];row=dict(method=method,development_accuracy=dev_accuracy,new_training_updates=7429 if method=='exact_native' else 0,original_updates=7429)
        for key in ['test','swapped','noisy']:row[key]=evaluate(model,fresh[key],'cuda')
        if route:route.restore()
        assert weights_sha(model.state_dict())==before;rows.append(row);references.append(dict(method=method,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),model_sha256=before))
    valid=all(rows[0][k]['accuracy']>=.99 for k in ['test','swapped'])
    for row in rows:row['gate']=valid and row['development_accuracy']>=.99 and all(row[k]['accuracy']>=.99 and row[k]['accuracy']>=rows[0][k]['accuracy']-.01 for k in ['test','swapped'])
    save(out/'reference-checkpoints.json',references);result=dict(status='complete',started_utc=started,finished_utc=now(),wall_seconds=time.perf_counter()-timer,main_updates=7429,indexer_updates=0,input_tokens=486400000,supervised_answers=30400000,technical_optimizer_updates=0,control_valid=valid,fit=fit,conditions=rows,scope='Exact full-QK native aggregation control, one seed and fixed 19-epoch dense-selected budget. No low-rank router, no speedup.');save(out/'result.json',result);save(out/'SUCCESS.json',dict(utc=now()));save(out/'progress.json',dict(utc=now(),status='complete'));save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(status='complete',control_valid=valid,conditions=[dict(method=row['method'],test=row['test']['accuracy'],swapped=row['swapped']['accuracy'],noisy=row['noisy']['accuracy'],gate=row['gate']) for row in rows])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    def interrupted(sig,frame):raise RuntimeError('Signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:main(args)
    except Exception:
        if args.output.exists():save(args.output/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()))
        raise
