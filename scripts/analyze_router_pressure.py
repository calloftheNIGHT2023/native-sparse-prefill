"""Replay frozen pressure results on CPU; verify lineage, steps, counts and gates."""
import json, math, shutil, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_frozen_router import now,save,sha,weights_sha,evaluate
from run_router_pressure import validate_data
from router_pressure_models import METHODS,load_checkpoint

def main():
    torch.set_num_threads(4);timer=time.perf_counter()
    src=ROOT/'results/router-pressure-v0';out=ROOT/'results/router-pressure-analysis-v0';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py')
    r=json.loads((src/'result.json').read_text());assert (src/'SUCCESS.json').exists() and not list(src.rglob('FAILURE.json'))
    manifest=json.loads((src/'manifest.json').read_text())
    for row in manifest:assert sha(src/row['path'])==row['sha256'],row['path']
    for p in (src/'source').glob('*.py'):assert sha(p)==sha(ROOT/'src'/p.name),p.name
    env=json.loads((src/'environment.json').read_text());assert sha(src/'preregistered-plan.md')==env['plan_sha256']==sha(ROOT/'docs/router-pressure-plan-2026-09-14.md')
    assert sha(ROOT/'data/router-pressure-kv16-v0/data.pt')==env['data_sha256']
    data=validate_data(ROOT/'data/router-pressure-kv16-v0');initial=torch.load(src/'initialization.pt',map_location='cpu',weights_only=True)
    T=r['common_epochs'];assert T in (40,80);decision=json.loads((src/'dense/budget-decision.json').read_text());assert decision['test_consulted'] is False and decision['common_epochs']==T
    dense40=torch.load(src/'dense/checkpoint_epoch040.pt',map_location='cpu',weights_only=False);assert dense40['epoch']==39 and dense40['steps']==12520
    assert T==(40 if decision['development_accuracy']>=.99 else 80)
    assert [x['method'] for x in r['runs']]==(list(METHODS) if r['runs'][0]['development_gate'] else ['dense'])
    rows=[];stamps=[];counts=0
    for run in r['runs']:
        name=run['method'];folder=src/name;learned=name in METHODS[1:3]
        ev=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()];steps=[x for x in ev if x['event']=='optimizer_step'];epochs=[x for x in ev if x['event']=='epoch'];conf=json.loads((folder/'config.json').read_text())
        assert conf['initial_backbone_sha']==weights_sha(initial['model'])
        assert conf['initial_indexer_sha']==(weights_sha(initial['indexers']) if learned else None)
        assert len(steps)==T*313 and [s['global_step'] for s in steps]==list(range(1,T*313+1))
        assert [x['epoch'] for x in epochs]==list(range(1,T+1)) and run['epochs']==T
        assert ev[0]['event']=='start' and ev[-1]['event']=='complete' and [e['utc'] for e in ev]==sorted(e['utc'] for e in ev)
        assert sum(s['batch_examples'] for s in steps)==T*10000
        assert sum(s['input_tokens'] for s in steps)==run['input_tokens']==T*1280000
        assert sum(s['supervised_answers'] for s in steps)==run['supervised_answers']==T*160000
        assert sum(s['indexer_update'] for s in steps)==run['indexer_updates']==(T*313 if learned else 0)
        assert run['main_updates']==T*313
        assert all(math.isfinite(s[k]) for s in steps for k in ['ce','aux','main_gradient_norm','indexer_gradient_norm','step_wall_seconds'])
        for epoch in range(1,T+1):
            ss=[s for s in steps if s['epoch']==epoch];assert [s['batch_examples'] for s in ss]==[32]*312+[16]
            assert all(math.isclose(s['main_lr'],.001*.5*(1+math.cos(math.pi*(epoch-1)/100)),abs_tol=1e-12) for s in ss)
        dense_slots=T*10000*2*128*128;sparse_slots=T*10000*2*128*8
        main_slots=dense_slots if name=='dense' else (10000*2*(4*128*128+(T-4)*128*8) if name.startswith('warm') else sparse_slots)
        expected=dict(main_score_entries=main_slots,index_score_entries=dense_slots if learned else 0,teacher_distribution_entries=main_slots if name.startswith('warm') else 0)
        assert run['score_accounting']==expected
        for k,v in expected.items():assert sum(s[k] for s in steps)==v
        assert math.isclose(sum(s['step_wall_seconds'] for s in steps),run['train_step_seconds'],rel_tol=1e-10)
        c=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert c['epoch']==T-1 and c['steps']==T*313 and c['epoch_boundary'] and c['scheduler']['last_epoch']==T
        assert all(float(v['step'])==T*313 for v in c['main_optimizer']['state'].values())
        if learned:assert all(float(v['step'])==T*313 for v in c['indexer_optimizer']['state'].values())
        else:assert c['indexer_optimizer'] is None
        m,ix,route=load_checkpoint(c,'cpu')
        for key,n in [('test',16384),('swapped',1024)]:
            actual=evaluate(m,data[key],'cpu');assert actual['predictions']==run[key]['predictions'],(name,key)
            assert actual['accuracy']==run[key]['accuracy'] and actual['answers']==run[key]['answers']==n;counts+=n
        dev=evaluate(m,data['development'],'cpu');assert dev['accuracy']==run['curves'][-1]['development_accuracy']
        train=evaluate(m,data['train'],'cpu')  # Post-fit descriptive fit/generalization check only.
        route.restore();assert weights_sha(m.state_dict())==run['final_backbone_sha'] and weights_sha(ix.state_dict())==run['final_indexer_sha']
        assert run['development_gate']==(epochs[-1]['development_accuracy']>=.99)
        if name=='dense':assert decision['development_accuracy']==epochs[39]['development_accuracy']
        first=next((x['epoch'] for x in epochs if x['development_accuracy']>=.99),None)
        rows.append(dict(method=name,test=run['test']['accuracy'],swapped=run['swapped']['accuracy'],first_dev99=first,final_dev=dev['accuracy'],final_training_accuracy=train['accuracy'],final_training_nll=train['nll'],gate=run.get('gate'),wall_seconds=run['wall_seconds'],train_step_seconds=run['train_step_seconds'],development_seconds=run['development_seconds'],checkpoint_seconds=run['checkpoint_seconds'],score_accounting=expected))
        stamps.append(dict(method=name,start=ev[0]['utc'],finish=ev[-1]['utc']))
    valid=r['runs'][0]['development_gate'] and all(r['runs'][0][key]['accuracy']>=.99 for key in ['test','swapped']);assert valid==r['control_valid']
    for run in r['runs'][1:]:assert run['gate']==(valid and all(run[key]['accuracy']>=.99 and run[key]['accuracy']>=r['runs'][0][key]['accuracy']-.01 for key in ['test','swapped']))
    for key in ['main_updates','indexer_updates','input_tokens','supervised_answers']:assert r[key]==sum(x[key] for x in r['runs'])
    summary=dict(status='audited',utc=now(),files_verified=len(manifest),all_cpu_predictions_match=True,answers_replayed=counts,development_answers_rechecked=16000*len(rows),postfit_training_answers_evaluated=160000*len(rows),analysis_optimizer_updates=0,common_epochs=T,control_valid=valid,rows=rows,training_stamps=stamps,analysis_wall_seconds=time.perf_counter()-timer,scope='Single seed synthetic MQAR pressure test of existing recipe adaptations. No novel method or real-text result.')
    save(out/'audit-and-summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4.3))
    for run in r['runs']:
        cs=run['curves'];ax[0].plot([c['epoch'] for c in cs],[100*c['development_accuracy'] for c in cs],label=run['method']);ax[1].plot([c['epoch'] for c in cs],[c['train_nll'] for c in cs])
    ax[0].set_ylim(0,102);ax[0].set_ylabel('Development accuracy (%)');ax[0].axhline(99,color='gray',ls=':');ax[0].legend(fontsize=7);ax[1].set_ylabel('Training NLL')
    for a in ax:a.set_xlabel('Epoch');a.grid(alpha=.2)
    fig.suptitle('KV16 | sequence length 128 | shared initialization | fixed common budget');fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);fig.savefig(out/'curves.pdf');plt.close(fig)
    save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(summary))

if __name__=='__main__':main()
