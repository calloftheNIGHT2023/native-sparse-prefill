"""CPU audit of every saved epoch dataset and frozen final-checkpoint predictions."""
import json,math,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_frozen_router import now,save,sha,weights_sha,evaluate
from router_pressure_models import load_checkpoint
from run_router_freshdata import validate,row_hashes,tensor_hash,generated

def main():
    torch.set_num_threads(4);timer=time.perf_counter();src=ROOT/'results/router-freshdata-v0';out=ROOT/'results/router-freshdata-analysis-v0';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py')
    r=json.loads((src/'result.json').read_text());assert r['status']=='complete' and (src/'SUCCESS.json').exists() and not (src/'FAILURE.json').exists()
    manifest=json.loads((src/'manifest.json').read_text())
    for row in manifest:assert sha(src/row['path'])==row['sha256'],row['path']
    for p in (src/'source').glob('*.py'):assert sha(p)==sha(ROOT/'src'/p.name)
    env=json.loads((src/'environment.json').read_text());assert sha(src/'plan.md')==env['plan_sha256']==sha(ROOT/'docs/router-freshdata-plan-2026-09-14.md')
    old=ROOT/'results/router-pressure-v0';assert sha(old/'dense/checkpoint.pt')==env['old_checkpoint_sha256'] and sha(old/'initialization.pt')==sha(src/'initialization.pt')==env['old_initialization_sha256']
    assert sha(old/'result.json')==env['old_result_sha256'] and sha(ROOT/'data/router-pressure-kv16-v0/data.pt')==env['old_data_sha256']
    initial=torch.load(src/'initialization.pt',map_location='cpu',weights_only=True);previous=torch.load(ROOT/'data/router-pressure-kv16-v0/data.pt',map_location='cpu',weights_only=True);new=torch.load(src/'new-test.pt',map_location='cpu',weights_only=True);validate(new['test']);assert tensor_hash(new['test'])==tensor_hash(generated(2026091591,1024))
    reserved=set();seen=set()
    for key in ['train','development','test','swapped']:reserved.update(row_hashes(previous[key]))
    assert not set(row_hashes(new['test']))&reserved
    for key in ['test','swapped']:reserved.update(row_hashes(new[key]))
    for i,(q,s,t) in enumerate(new['swap_positions']):
        x=new['test']['inputs'][i];z=new['swapped']['inputs'][i];y=new['swapped']['labels'][i];assert int((x!=z).sum())==2 and torch.equal(x[32:],z[32:]);assert int(y[q])==int(x[t])!=int(x[s]) and int((y!=-100).sum())==1
    data_meta=json.loads((src/'data-seeds.json').read_text());assert [x['seed'] for x in data_meta]==list(range(2026091501,2026091581))
    for row in data_meta:
        epoch=row['epoch'];arr=np.load(src/'training-data'/f'epoch{epoch:03d}.npz',allow_pickle=False);data={k:torch.from_numpy(arr[k]) for k in ['inputs','labels']};validate(data)
        assert tensor_hash(data)==row['tensor_sha256'];hashes=set(row_hashes(data));assert len(hashes)==10000 and not hashes&reserved and not hashes&seen;seen.update(hashes)
        if epoch in (1,80):assert tensor_hash(generated(row['seed'],10000))==row['tensor_sha256']
    assert len(seen)==r['unique_training_sequences']==800000
    events=[json.loads(x) for x in (src/'events.jsonl').read_text().splitlines()];steps=[x for x in events if x['event']=='optimizer_step'];epochs=[x for x in events if x['event']=='epoch'];assert events[0]['event']=='start' and events[-1]['event']=='fit_complete' and [x['utc'] for x in events]==sorted(x['utc'] for x in events)
    assert events[0]['initial_backbone_sha']==weights_sha(initial['model']);assert [x['global_step'] for x in steps]==list(range(1,25041));assert [x['epoch'] for x in epochs]==list(range(1,81))
    assert [dict(epoch=x['epoch'],seed=x['seed'],tensor_sha256=x['tensor_sha256'],rows=x['rows'],new_unique_rows=x['new_unique_rows']) for x in events if x['event']=='data_generated']==data_meta
    for epoch in range(1,81):
        ss=[s for s in steps if s['epoch']==epoch];assert [s['batch_examples'] for s in ss]==[32]*312+[16];assert all(math.isclose(s['main_lr'],.001*.5*(1+math.cos(math.pi*(epoch-1)/100)),abs_tol=1e-12) for s in ss)
    assert all(math.isfinite(x[k]) for x in steps for k in ['ce','main_gradient_norm','step_wall_seconds'])
    assert sum(s['input_tokens'] for s in steps)==r['input_tokens']==102400000 and sum(s['supervised_answers'] for s in steps)==r['supervised_answers']==12800000
    assert r['main_updates']==25040 and r['indexer_updates']==0;assert math.isclose(sum(s['step_wall_seconds'] for s in steps),r['train_step_seconds'],rel_tol=1e-10)
    c=torch.load(src/'checkpoint.pt',map_location='cpu',weights_only=False);assert c['epoch']==79 and c['steps']==25040 and c['epoch_boundary'] and c['scheduler']['last_epoch']==80 and c['indexer_optimizer'] is None
    assert all(float(v['step'])==25040 for v in c['main_optimizer']['state'].values());assert weights_sha(c['model'])==r['final_backbone_sha']
    for epoch in (20,40,80):
        ck=torch.load(src/f'checkpoint_epoch{epoch:03d}.pt',map_location='cpu',weights_only=False);assert ck['epoch']==epoch-1 and ck['steps']==epoch*313 and ck['epoch_boundary']
    rows=[];replayed=0;extra=0
    for label,path in [('fixed_data_epoch80',old/'dense/checkpoint.pt'),('fresh_data_epoch40',src/'checkpoint_epoch040.pt'),('fresh_data_epoch80',src/'checkpoint.pt')]:
        ck=torch.load(path,map_location='cpu',weights_only=False);m,ix,route=load_checkpoint(ck,'cpu');expected=next((e for e in r['evaluations'] if e['condition']==label),None);row=dict(condition=label,checkpoint_sha256=sha(path))
        for key in ['test','swapped']:
            actual=evaluate(m,new[key],'cpu');row[key]=actual['accuracy']
            if expected:assert actual['predictions']==expected[key]['predictions'] and actual['accuracy']==expected[key]['accuracy'];replayed+=actual['answers']
            else:extra+=actual['answers']
            if key=='test':
                p=torch.tensor(actual['predictions']).view(1024,16);bank=new[key]['inputs'][:,1:32:2];row['prediction_in_source_value_bank']=float((p[:,:,None]==bank[:,None,:]).any(-1).double().mean())
        if label=='fresh_data_epoch80':
            dev=evaluate(m,previous['development'],'cpu');assert dev['accuracy']==epochs[-1]['development_accuracy'];row['development_accuracy']=dev['accuracy'];last=evaluate(m,data,'cpu');row['last_training_epoch_accuracy']=last['accuracy'];row['last_training_epoch_nll']=last['nll'];extra+=dev['answers']+last['answers']
        route.restore();rows.append(row)
    assert r['gate']==(rows[-1]['development_accuracy']>=.99 and rows[-1]['test']>=.99 and rows[-1]['swapped']>=.99)
    summary=dict(status='audited',utc=now(),files_verified=len(manifest),unique_training_rows_verified=len(seen),data_epochs_verified=80,first_and_last_epoch_regenerated_on_cpu=True,answers_replayed=replayed,additional_diagnostic_answers=extra,optimizer_updates=0,gate=r['gate'],rows=rows,first_development_99=next((x['epoch'] for x in epochs if x['development_accuracy']>=.99),None),training_start_utc=events[0]['utc'],training_end_utc=events[-1]['utc'],analysis_wall_seconds=time.perf_counter()-timer)
    save(out/'audit-and-summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.2));old_r=json.loads((old/'result.json').read_text())
    for label,cs in [('Repeat 10,000 fixed examples',old_r['runs'][0]['curves']),('10,000 fresh examples per epoch',r['curves'])]:
        axes[0].plot([x['epoch'] for x in cs],[x['development_accuracy']*100 for x in cs],label=label);axes[1].plot([x['epoch'] for x in cs],[x['train_nll'] for x in cs],label=label)
    axes[0].set_ylabel('Development accuracy (%)');axes[0].set_ylim(0,102);axes[0].axhline(99,ls=':',color='gray');axes[0].legend(fontsize=8);axes[1].set_ylabel('Online training NLL')
    for ax in axes:ax.set_xlabel('Epoch (10,000 training exposures)');ax.grid(alpha=.2)
    fig.suptitle('Exploratory dense KV16 control | same initialization, 80 epochs and optimizer');fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);fig.savefig(out/'curves.pdf');plt.close(fig)
    save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(summary))

if __name__=='__main__':main()
