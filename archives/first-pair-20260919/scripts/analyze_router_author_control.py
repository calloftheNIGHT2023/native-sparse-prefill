"""Verify author-config provenance, all scientific updates and held-out GPU predictions."""
import json,math,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from router_author_control import author_configs,make_model,validate,row_hashes,evaluate,LRS
from run_frozen_router import now,save,sha,weights_sha

def main():
    torch.set_num_threads(4);timer=time.perf_counter();src=ROOT/'results/router-author-control-v0';out=ROOT/'results/router-author-analysis-v0';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py')
    r=json.loads((src/'result.json').read_text());assert (src/'SUCCESS.json').exists() and not list(src.rglob('FAILURE.json'))
    manifest=json.loads((src/'manifest.json').read_text())
    for row in manifest:assert sha(src/row['path'])==row['sha256'],row['path']
    env=json.loads((src/'environment.json').read_text());assert env['plan_sha256']==sha(src/'plan.md')==sha(ROOT/'docs/router-author-control-plan-2026-09-14.md')
    for p in (src/'source').glob('*.py'):
        if p.name!='author-paper-configs.py':assert sha(p)==sha(ROOT/'src'/p.name),p.name
    assert sha(src/'source/author-paper-configs.py')==env['author_source_sha256']
    expected_cfgs=author_configs();assert json.loads((src/'resolved-configs.json').read_text())==expected_cfgs
    data=torch.load(src/'data.pt',map_location='cpu',weights_only=True);audit=json.loads((src/'data-audit.json').read_text());assert sha(src/'data.pt')==audit['data_sha256'];initial=torch.load(src/'initialization.pt',map_location='cpu',weights_only=True);assert weights_sha(initial)==audit['initial_backbone_sha']
    seen=set()
    for key,n in [('train',100000),('development',3000),('test',1024)]:
        assert len(data[key]['inputs'])==n;validate(data[key]);hashes=row_hashes(data[key]);assert len(hashes)==n and not hashes&seen;seen.update(hashes)
    validate(data['noisy'],False);assert torch.equal(data['noisy']['inputs'][:,:32],data['test']['inputs'][:,:32]) and torch.equal(data['noisy']['labels'],data['test']['labels'])
    noise_mask=data['test']['inputs']==0;fill=torch.randint(0,8192,data['test']['inputs'].shape,generator=torch.Generator().manual_seed(2026091602));expected=data['test']['inputs'].clone();expected[noise_mask]=fill[noise_mask];assert torch.equal(expected,data['noisy']['inputs'])
    for i,(q,s,t) in enumerate(data['swap_positions']):
        x=data['test']['inputs'][i];z=data['swapped']['inputs'][i];y=data['swapped']['labels'][i];assert int((x!=z).sum())==2 and torch.equal(x[32:],z[32:]);assert int((y!=-100).sum())==1 and int(y[q])==int(x[t])!=int(x[s])
    rng=np.random.RandomState(123);assert audit['seeds']['train']==int(rng.randint(0,2**31,size=1,dtype=np.int64)[0]);assert audit['seeds']['development']==int(rng.randint(2**31,2**32,size=1,dtype=np.int64)[0]);assert audit['seeds']['test']==2026091601 and audit['seeds']['paired_noise']==2026091602
    rows=[];stamps=[];answers=0
    for i,run in enumerate(r['runs']):
        folder=src/run['method'];E=run['epochs'];cfg=json.loads((folder/'config.json').read_text());assert cfg['config']==expected_cfgs[i] and cfg['initial_backbone_sha']==weights_sha(initial);assert run['learning_rate']==LRS[i]
        events=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()];steps=[s for s in events if s['event']=='optimizer_step'];epochs=[s for s in events if s['event']=='epoch'];assert events[0]['event']=='start' and events[-1]['event']=='fit_complete' and [e['utc'] for e in events]==sorted(e['utc'] for e in events)
        assert [s['global_step'] for s in steps]==list(range(1,E*391+1));assert [s['epoch'] for s in epochs]==list(range(1,E+1));assert [s['early_stop'] for s in epochs[:-1]]==[False]*(E-1)
        assert run['development_gate']==(epochs[-1]['development_accuracy']>.99) and (run['development_gate'] or E==64)
        for epoch in range(1,E+1):
            ss=[s for s in steps if s['epoch']==epoch];assert [s['batch_examples'] for s in ss]==[256]*390+[160]
            assert all(math.isclose(s['learning_rate'],LRS[i]*.5*(1+math.cos(math.pi*(epoch-1)/64)),abs_tol=1e-12) for s in ss)
        assert sum(s['input_tokens'] for s in steps)==run['input_tokens']==E*25600000;assert sum(s['supervised_answers'] for s in steps)==run['supervised_answers']==E*1600000;assert run['main_updates']==len(steps) and run['indexer_updates']==0
        assert all(math.isfinite(s[k]) for s in steps for k in ['loss','gradient_norm','train_step_seconds']);assert math.isclose(sum(s['train_step_seconds'] for s in steps),run['train_step_seconds'],rel_tol=1e-10)
        c=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert c['epoch']==E-1 and c['steps']==E*391 and c['epoch_boundary'];assert c['scheduler']['last_epoch']==E-int(run['development_gate']);assert all(float(v['step'])==E*391 for v in c['optimizer']['state'].values())
        m=make_model(c['config'],c['model']);assert sum(p.numel() for p in m.parameters())==1214720 and m.lm_head.weight is m.backbone.embeddings.word_embeddings.weight
        for key,n in [('test',16384),('swapped',1024),('noisy',16384)]:
            actual=evaluate(m,data[key],'cpu');assert actual['predictions']==run[key]['predictions'] and actual['accuracy']==run[key]['accuracy'] and actual['answers']==n;answers+=n
        dev=evaluate(m,data['development'],'cpu');assert dev['accuracy']==epochs[-1]['development_accuracy'];assert weights_sha(m.state_dict())==run['final_backbone_sha']
        assert run['gate']==(run['development_gate'] and run['test']['accuracy']>=.99 and run['swapped']['accuracy']>=.99)
        targets=data['test']['labels'][data['test']['labels']!=-100].reshape(1024,16).numpy();clean=np.array(run['test']['predictions']).reshape(1024,16);noisy=np.array(run['noisy']['predictions']).reshape(1024,16);delta=((clean==targets).mean(-1)-(noisy==targets).mean(-1));rng=np.random.default_rng(2026091614);bootstrap=[]
        for _ in range(10):bootstrap.extend(delta[rng.integers(0,1024,(500,1024))].mean(-1).tolist())
        interval=np.quantile(bootstrap,[.025,.975]).tolist()
        rows.append(dict(method=run['method'],learning_rate=LRS[i],epochs=E,development=dev['accuracy'],test=run['test']['accuracy'],swapped=run['swapped']['accuracy'],noisy=run['noisy']['accuracy'],gate=run['gate'],train_seconds=run['train_step_seconds'],wall_seconds=run['wall_seconds'],parameters=1214720,noise_accuracy_drop=float(delta.mean()),paired_sequence_bootstrap95_drop=interval,noise_prediction_changed=float((clean!=noisy).mean())));stamps.append(dict(method=run['method'],start=events[0]['utc'],finish=events[-1]['utc']))
    assert len(r['runs'])==(1 if r['runs'][0]['development_gate'] else 2);assert r['skipped_learning_rates']==list(LRS[len(r['runs']):])
    assert r['control_valid']==any(x['gate'] for x in r['runs'])
    for key in ['main_updates','input_tokens','supervised_answers']:assert r[key]==sum(x[key] for x in r['runs'])
    summary=dict(status='audited',utc=now(),files_verified=len(manifest),rows=rows,all_gpu_predictions_reproduced=True,answers_replayed=answers,development_answers_rechecked=48000*len(rows),optimizer_updates=0,training_stamps=stamps,control_valid=r['control_valid'],analysis_wall_seconds=time.perf_counter()-timer);save(out/'audit-and-summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.1))
    for run in r['runs']:
        cs=run['curves'];label=f"lr={run['learning_rate']:.5g}";ax[0].plot([x['epoch'] for x in cs],[x['development_accuracy']*100 for x in cs],label=label);ax[1].plot([x['epoch'] for x in cs],[x['train_nll'] for x in cs],label=label)
    ax[0].set_ylim(0,102);ax[0].set_ylabel('Development accuracy (%)');ax[0].axhline(99,color='gray',ls=':');ax[0].legend();ax[1].set_ylabel('Training NLL')
    for a in ax:a.set_xlabel('Epoch (100,000 examples)');a.grid(alpha=.2)
    fig.suptitle('Pinned author configuration | dense attention | KV16 / length256 / vocab8192');fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);fig.savefig(out/'curves.pdf');plt.close(fig)
    save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(summary))

if __name__=='__main__':main()
