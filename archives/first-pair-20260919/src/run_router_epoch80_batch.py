"""Matched-budget continuation, with a new test held until all three fits finish."""
import argparse,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import torch
from zoology_entry import ROOT,UPSTREAM,multiquery_ar
from resume_joint_token_router import restore_checkpoint
from run_frozen_router import now,save,sha,evaluate,value_swap

def main(a):
    a.output.mkdir(parents=True,exist_ok=False);started=now();timer=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=ROOT/'docs/joint-token-continuation-plan-2026-09-14.md';shutil.copy2(plan,a.output/'preregistered-plan.md');source=ROOT/'results/joint-token-router-v0';data_file=source/'data-and-initialization.pt'
    snapshot=a.output/'source';snapshot.mkdir()
    for name in ['run_router_epoch80_batch.py','resume_joint_token_router.py','run_joint_token_router.py','run_frozen_router.py','frozen_routing.py','joint_token_routing.py','zoology_entry.py','zoology_sparse_schedule.py']:shutil.copy2(ROOT/'src'/name,snapshot/name)
    shutil.copytree(UPSTREAM,snapshot/'upstream',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    old=torch.load(data_file,map_location='cpu',weights_only=True);previous=[old[k]['inputs'] for k in ['train','development','fresh','swapped']]
    frozen=torch.load(ROOT/'results/frozen-router-capacity-v0/data.pt',map_location='cpu',weights_only=True);previous += [frozen['fresh']['inputs'],frozen['swapped']['inputs']]
    hist=torch.load(ROOT/'results/schedule-screen-cloud-v0/shared-data-and-initialization.pt',map_location='cpu',weights_only=True);previous+=[d['inputs'] for d in hist['splits'].values()]
    torch.manual_seed(2026091481);seg=multiquery_ar(vocab_size=256,num_examples=1024,input_seq_len=64,seed=2026091481,num_kv_pairs=4);fresh=dict(inputs=seg.inputs,labels=seg.labels);swapped,pairs=value_swap(fresh)
    oldhash={hashlib.sha256(row.numpy().tobytes()).hexdigest() for t in previous for row in t};newhash={hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in fresh['inputs']};assert len(newhash)==1024 and not oldhash&newhash
    for x,y in zip(fresh['inputs'],fresh['labels']):
        mapping={int(x[i]):int(x[i+1]) for i in range(0,8,2)};assert len(mapping)==4 and int((y!=-100).sum())==4
        for pos in torch.where(y!=-100)[0]:assert mapping[int(x[pos])]==int(y[pos])
    torch.save(dict(fresh=fresh,swapped=swapped,swap_positions=pairs),a.output/'new-test.pt');save(a.output/'data-audit.json',dict(utc=now(),seed=2026091481,rows=1024,answers=4096,swap_answers=1024,disjoint_old_rows=True,labels_verified=True))
    save(a.output/'environment.json',dict(utc=started,torch=str(torch.__version__),python=sys.version,device=a.device,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,plan_sha256=sha(plan)))
    completed=[]
    for name in ['exact_r16_shadow','learned_r16','learned_r64']:
        left=2700-(time.perf_counter()-timer)
        if left<30:raise TimeoutError('Overall 45-minute cap')
        limit=min(720,int(left)-20);folder=a.output/name;console=a.output/(name+'-console.log')
        cmd=[sys.executable,'-u',str(ROOT/'src/resume_joint_token_router.py'),'--checkpoint',str(source/name/'checkpoint.pt'),'--data',str(data_file),'--output',str(folder),'--plan',str(plan),'--total-epochs','80','--max-seconds',str(limit),'--device',a.device]
        print(json.dumps(dict(utc=now(),event='starting',method=name)),flush=True)
        with console.open('w',encoding='utf-8') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=limit+15)
        r=json.loads((folder/'result.json').read_text());r['method']=name;assert r['additional_backbone_updates']==12520 and r['additional_indexer_updates']==12520;completed.append(r);save(a.output/'progress.json',dict(utc=now(),status='fitting',completed=[x['method'] for x in completed]));print(json.dumps(dict(utc=now(),event='fit_complete',method=name)),flush=True)
    for r in completed:
        ck=torch.load(a.output/r['method']/'checkpoint.pt',map_location='cpu',weights_only=False);m,ix,route,opt,iopt,sch=restore_checkpoint(ck,a.device);m.eval();route.collect_aux=False
        r.update(fresh=evaluate(m,fresh,a.device),swapped=evaluate(m,swapped,a.device));route.restore();r['status']='complete';save(a.output/r['method']/'final-evaluation.json',r)
    control=completed[0];control_valid=all(control[k]['accuracy']>=.99 for k in ['fresh','swapped'])
    for r in completed[1:]:r['epoch80_gate']=control_valid and all(r[k]['accuracy']>=.99 and r[k]['accuracy']>=control[k]['accuracy']-.01 for k in ['fresh','swapped']);save(a.output/r['method']/'final-evaluation.json',r)
    result=dict(status='complete',started_utc=started,finished_utc=now(),wall_seconds=time.perf_counter()-timer,additional_backbone_updates=37560,additional_indexer_updates=37560,additional_input_tokens=76800000,additional_supervised_answers=4800000,control_valid=control_valid,runs=completed,scope='Conditional 40-to-80 epoch continuation on a new A40; same single seed, new held-out data, no originality or real-LLM conclusion')
    save(a.output/'result.json',result);save(a.output/'SUCCESS.json',dict(utc=now()));save(a.output/'progress.json',dict(utc=now(),status='complete'));save(a.output/'manifest.json',[dict(path=p.relative_to(a.output).as_posix(),sha256=sha(p)) for p in sorted(a.output.rglob('*')) if p.is_file()]);print(json.dumps(dict(status='complete',runs=[dict(method=r['method'],fresh=r['fresh']['accuracy'],swapped=r['swapped']['accuracy'],gate=r.get('epoch80_gate')) for r in completed])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    try:main(a)
    except Exception:
        if a.output.exists():save(a.output/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()))
        raise
