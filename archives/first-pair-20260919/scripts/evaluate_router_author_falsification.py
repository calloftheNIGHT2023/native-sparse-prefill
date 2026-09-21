"""One frozen fresh evaluation and independent CPU prediction replay."""
import argparse,json,sys,shutil,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from zoology_entry import ROOT
from router_author_control import author_configs,make_model,generated,validate,row_hashes,interventions,evaluate
from run_router_author_falsification import model_for
from run_frozen_router import now,save,sha,weights_sha

def main(a):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.output.resolve();cfg=author_configs()[1]
    if a.replay:
        old=json.loads((out/'evaluation.json').read_text(encoding='utf-8'));fresh=torch.load(out/'evaluation-data.pt',map_location='cpu',weights_only=True);checked=[]
        for row in old['conditions']:
            p=ROOT/row['checkpoint'];assert sha(p)==row['checkpoint_sha256'];c=torch.load(p,map_location='cpu',weights_only=False);m=model_for(cfg,c['model'],'cpu',row['method'])
            assert weights_sha(m.state_dict())==row['model_sha256']
            for key in ['test','swapped','noisy']:
                v=evaluate(m,fresh[key],'cpu',batch=32);assert v['predictions']==row[key]['predictions'];assert v['accuracy']==row[key]['accuracy'];checked.append(dict(condition=row['name'],split=key,answers=v['answers'],predictions_equal=True))
            print(json.dumps({'replayed':row['name']}),flush=True)
        save(out/'cpu-replay.json',dict(utc=now(),status='passed',answers=sum(x['answers'] for x in checked),checked=checked,optimizer_updates=0));return
    out.mkdir(parents=True,exist_ok=False);started=now();tick=time.perf_counter()
    # Freeze list and checkpoint hashes before any fresh predictions are generated.
    spec=json.loads(a.spec.read_text(encoding='utf-8'));references=[]
    for s in spec:
        p=ROOT/s['checkpoint'];references.append(dict(**s,checkpoint_sha256=sha(p)))
    save(out/'frozen-references.json',dict(utc=now(),references=references));shutil.copy2(__file__,out/'evaluation-source.py')
    fresh_test=generated(2026091701,1024);validate(fresh_test);hh=row_hashes(fresh_test);assert len(hh)==1024
    prior=torch.load(ROOT/'results/router-author-control-v0/data.pt',map_location='cpu',weights_only=True)
    compared=[]
    for key in ['train','development','test']:
        assert not hh&row_hashes(prior[key]);compared.append('router-author-control-v0/'+key)
    for name in ['router-author-sparse-v0','router-author-exact-train-v0']:
        dd=torch.load(ROOT/'results'/name/'evaluation-data.pt',map_location='cpu',weights_only=True);assert not hh&row_hashes(dd['test']);compared.append(name+'/test')
    swapped,noisy,pairs=interventions(fresh_test);mask=fresh_test['inputs']==0;fill=torch.randint(0,8192,fresh_test['inputs'].shape,generator=torch.Generator().manual_seed(2026091702));noisy['inputs']=fresh_test['inputs'].clone();noisy['inputs'][mask]=fill[mask];validate(noisy,False)
    for i,(q,s,t) in enumerate(pairs):
        assert swapped['labels'][i,q]==fresh_test['inputs'][i,t]
        assert swapped['inputs'][i,s]==fresh_test['inputs'][i,t] and swapped['inputs'][i,t]==fresh_test['inputs'][i,s]
        assert swapped['labels'][i,q]!=fresh_test['labels'][i,q]
    fresh=dict(test=fresh_test,swapped=swapped,noisy=noisy,swap_positions=pairs);torch.save(fresh,out/'evaluation-data.pt')
    save(out/'data-audit.json',dict(utc=now(),test_seed=2026091701,noise_seed=2026091702,new_rows=True,compared=compared,clean_and_noise_labels_valid=True,all_swap_labels_changed=True,data_sha256=sha(out/'evaluation-data.pt')))
    rows=[]
    for ref in references:
        p=ROOT/ref['checkpoint'];c=torch.load(p,map_location='cpu',weights_only=False);assert c['epoch_boundary'];m=model_for(cfg,c['model'],'cuda',ref['method']);before=weights_sha(m.state_dict());row=dict(**ref,model_sha256=before,epoch=c['epoch']+1,steps=c['steps'])
        for key in ['test','swapped','noisy']:row[key]=evaluate(m,fresh[key],'cuda')
        assert sha(p)==ref['checkpoint_sha256'] and before==weights_sha(m.state_dict());row['clean_and_swap_gate']=all(row[k]['accuracy']>=.99 for k in ['test','swapped']);rows.append(row)
        print(json.dumps(dict(name=row['name'],epoch=row['epoch'],**{k:row[k]['accuracy'] for k in ['test','swapped','noisy']})),flush=True)
        del m
    save(out/'evaluation.json',dict(started_utc=started,finished_utc=now(),wall_seconds=time.perf_counter()-tick,conditions=rows,optimizer_updates=0,checkpoint_selection='final frozen checkpoint for each condition; no test selection'))
    save(out/'SUCCESS.json',dict(utc=now()));save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--spec',type=Path);p.add_argument('--replay',action='store_true');main(p.parse_args())
