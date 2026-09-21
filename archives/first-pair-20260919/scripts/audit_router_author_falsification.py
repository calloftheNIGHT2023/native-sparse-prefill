"""Audit new update counts, immutable sources, checkpoint state and CPU replay."""
import argparse,json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from zoology_entry import ROOT
from run_frozen_router import now,save,sha,weights_sha

def main(a):
    dirs=[p.resolve() for p in a.fits];rows=[]
    for d in dirs:
        assert (d/'SUCCESS.json').exists() and not (d/'FAILURE.json').exists();fit=json.loads((d/'fit-result.json').read_text(encoding='utf-8'));prov=json.loads((d/'provenance.json').read_text(encoding='utf-8'))
        assert sha(d/'plan.md')==prov['plan_sha256']
        for ref in json.loads((d/'source-manifest.json').read_text(encoding='utf-8')):assert sha(d/'source'/ref['path'])==ref['sha256']
        # Parent absolute paths were cloud paths; translate the project-relative suffix.
        for path,h in prov['protected'].items():
            relative=path.split('/router-study-20260914-v1/')[-1];assert sha(ROOT/relative)==h
        n=0;tokens=0;answers=0;epochs=[];first=fit['prefix_main_updates'];last_lr=None
        for line in (d/'events.jsonl').read_text(encoding='utf-8').splitlines():
            e=json.loads(line)
            if e['event']=='optimizer_step':
                n+=1;assert e['global_step']==first+n and e['new_main_updates']==n;assert e['indexer_updates']==0;assert math.isfinite(e['ce']) and math.isfinite(e['main_gradient_norm']);tokens+=e['input_tokens'];answers+=e['supervised_answers'];last_lr=e['main_lr']
                expected_lr=fit['lr']*(1+math.cos(math.pi*(e['epoch']-1)/64))/2
                assert math.isclose(e['main_lr'],expected_lr,rel_tol=1e-11,abs_tol=1e-15)
            elif e['event']=='epoch':epochs.append(e)
        assert n==fit['new_main_updates']==(64-fit['starting_epoch'])*391;assert tokens==fit['input_tokens'];assert answers==fit['supervised_answers'];assert len(epochs)==64-fit['starting_epoch']
        ck=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch_boundary'] and ck['epoch']==63 and ck['steps']==25024;assert ck['scheduler']['last_epoch']==64 and ck['scheduler']['T_max']==64;assert weights_sha(ck['model'])==fit['final_model_sha256'];assert ck['optimizer']['param_groups'][0]['lr']==0.0
        assert ck['scheduler']['base_lrs']==[fit['lr']]
        save(d/'effective-config.json',dict(documentation_created_utc=now(),post_training_documentation_only=True,author_template_sha256=sha(d/'config.json'),model=ck['config']['model'],method=fit['method'],initialization_and_training_seed=fit['seed'],data_master_seed=123,training_generation_seed=843828734,development_generation_seed=3062119789,training_data_reference='results/router-author-control-v0/data.pt',batch_size=256,fixed_training_order=True,optimizer=dict(name='AdamW',initial_learning_rate=fit['lr'],weight_decay=.1),scheduler=dict(name='CosineAnnealingLR',T_max=64),starting_epoch=fit['starting_epoch'],final_epoch=64,early_stop=False,gradient_clipping=False,tf32=False,compute_dtype='float32',actual_learning_rate_verified_for_every_update=True))
        save(d/'manifest.json',[dict(path=p.relative_to(d).as_posix(),sha256=sha(p)) for p in sorted(d.rglob('*')) if p.is_file() and p.name!='manifest.json'])
        rows.append(dict(directory=d.relative_to(ROOT).as_posix(),main_updates=n,indexer_updates=0,input_tokens=tokens,supervised_answers=answers,epochs=len(epochs),final_epoch=64,last_used_lr=last_lr,final_development_accuracy=fit['curves'][-1]['development_accuracy'],wall_seconds=fit['wall_seconds'],first_crossing=fit['first_observed_development_crossing'],checkpoint_sha256=sha(d/'checkpoint.pt')))
    replay=json.loads((a.evaluation/'cpu-replay.json').read_text(encoding='utf-8'));assert replay['status']=='passed'
    out=a.output.resolve();out.parent.mkdir(parents=True,exist_ok=True);save(out,dict(utc=now(),status='passed',fits=rows,main_updates=sum(r['main_updates'] for r in rows),indexer_updates=0,input_tokens=sum(r['input_tokens'] for r in rows),supervised_answers=sum(r['supervised_answers'] for r in rows),cpu_replayed_answers=replay['answers'],notes='Prefix updates excluded. Disposable resume tests separately logged. Source hashes and parent hashes checked.'))
    print(out.read_text(encoding='utf-8'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fits',type=Path,nargs='+',required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
