"""Recover the final reference-path serialization only. Never construct an optimizer or update weights."""
import json,shutil,time
from pathlib import Path
from datetime import datetime
import torch
from zoology_entry import ROOT
from router_author_control import make_model,evaluate
from router_author_sparse import build,METHODS
from run_router_author_exact_train import exact_model
from run_frozen_router import now,save,sha,weights_sha

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
out=(ROOT/'results/router-author-exact-train-v0').resolve();old=ROOT/'results/router-author-control-v0';prior=ROOT/'results/router-author-sparse-v0';assert out.is_relative_to(ROOT.resolve()) and (out/'FAILURE.json').exists() and not (out/'SUCCESS.json').exists();error=json.loads((out/'FAILURE.json').read_text());assert 'path.relative_to(ROOT)' in error['traceback'] and 'ValueError' in error['traceback'];assert not (out/'failure-checkpoint.pt').exists()
fit=json.loads((out/'fit-result.json').read_text());cfg=json.loads((out/'config.json').read_text());ck=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);assert fit['epochs']==19 and fit['main_updates']==7429 and ck['epoch']==18 and ck['steps']==7429 and ck['epoch_boundary'] and ck['scheduler']['last_epoch']==19;assert all(float(v['step'])==7429 for v in ck['optimizer']['state'].values());assert weights_sha(ck['model'])==fit['final_backbone_sha'];events=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()];assert events[-1]['event']=='fit_complete' and len([e for e in events if e['event']=='optimizer_step'])==7429
protected={p.name:sha(p) for p in out.glob('checkpoint*.pt')};protected.update({n:sha(out/n) for n in ['initialization.pt','events.jsonl','fit-result.json','evaluation-data.pt','data-audit.json','config.json','plan.md']});started=now();timer=time.perf_counter();data=torch.load(out/'evaluation-data.pt',map_location='cpu',weights_only=True);shutil.copy2(__file__,out/'source/recover_router_author_exact_metadata.py');rows=[];references=[]
for method in ['dense_existing',*METHODS,'exact_native']:
    route=None
    if method=='exact_native':path=out/'checkpoint.pt';source_fit=fit;c=ck;m=exact_model(cfg,c['model'],'cuda')
    elif method=='dense_existing':
        path=old/'lr2/checkpoint.pt';c=torch.load(path,map_location='cpu',weights_only=False);m=make_model(cfg,c['model'],'cuda');source_fit=json.loads((old/'lr2/fit-result.json').read_text())
    else:
        path=prior/method/'checkpoint.pt';c=torch.load(path,map_location='cpu',weights_only=False);m,ix,route=build(cfg,c['model'],c['indexers'],method,'cuda');route.epoch=19;route.collect_aux=False;source_fit=json.loads((prior/method/'fit-result.json').read_text())
    before=weights_sha(m.state_dict());assert before==source_fit['final_backbone_sha'];row=dict(method=method,development_accuracy=source_fit['curves'][-1]['development_accuracy'],new_training_updates=7429 if method=='exact_native' else 0,original_updates=7429)
    for key in ['test','swapped','noisy']:row[key]=evaluate(m,data[key],'cuda')
    if route:route.restore()
    assert weights_sha(m.state_dict())==before;rows.append(row);references.append(dict(method=method,path=path.resolve().relative_to(ROOT.resolve()).as_posix(),sha256=sha(path),model_sha256=before))
valid=all(rows[0][k]['accuracy']>=.99 for k in ['test','swapped'])
for row in rows:row['gate']=valid and row['development_accuracy']>=.99 and all(row[k]['accuracy']>=.99 and row[k]['accuracy']>=rows[0][k]['accuracy']-.01 for k in ['test','swapped'])
for name,h in protected.items():assert sha(out/name)==h,name
original_started=json.loads((out/'environment.json').read_text())['utc'];finished=now();recovery=dict(started_utc=started,finished_utc=finished,wall_seconds=time.perf_counter()-timer,optimizer_updates=0,protected_files_unchanged=protected,original_error='Relative output checkpoint path could not be serialized relative to absolute project root after the training and first evaluation pass completed.',evaluation_repeated_with_frozen_weights=True,source_sha256=sha(Path(__file__)))
result=dict(status='complete',started_utc=original_started,finished_utc=finished,wall_seconds=(datetime.fromisoformat(finished)-datetime.fromisoformat(original_started)).total_seconds(),wall_seconds_includes_metadata_recovery_gap=True,main_updates=7429,indexer_updates=0,input_tokens=486400000,supervised_answers=30400000,technical_optimizer_updates=0,control_valid=valid,fit=fit,conditions=rows,recovered_metadata_only=True,scope='Exact full-QK native aggregation control. Frozen checkpoint evaluation repeated after metadata-path failure, with zero additional updates.')
save(out/'recovery.json',recovery);save(out/'reference-checkpoints.json',references);save(out/'result.json',result);(out/'FAILURE.json').rename(out/'RECOVERED_FAILURE.json');save(out/'SUCCESS.json',dict(utc=finished,recovered_metadata_only=True));save(out/'progress.json',dict(utc=finished,status='complete',recovered_metadata_only=True));save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(status='complete',control_valid=valid,recovery_optimizer_updates=0,conditions=[dict(method=row['method'],test=row['test']['accuracy'],swapped=row['swapped']['accuracy'],noisy=row['noisy']['accuracy'],gate=row['gate']) for row in rows])))
