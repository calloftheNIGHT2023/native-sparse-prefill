"""Audit frozen 19-epoch fits, update counts, score accounting and all GPU predictions."""
import json,math,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_frozen_router import now,save,sha,weights_sha
from router_author_control import make_model,evaluate,validate,row_hashes,author_configs
from router_author_sparse import build,METHODS

def main():
    timer=time.perf_counter();torch.set_num_threads(4);src=ROOT/'results/router-author-sparse-v0';old=ROOT/'results/router-author-control-v0';out=ROOT/'results/router-author-sparse-analysis-v0';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py')
    result=json.loads((src/'result.json').read_text(encoding='utf-8'));assert (src/'SUCCESS.json').exists() and not list(src.rglob('FAILURE.json'));manifest=json.loads((src/'manifest.json').read_text(encoding='utf-8'))
    for row in manifest:assert sha(src/row['path'])==row['sha256'],row['path']
    for path in (src/'source').glob('*.py'):assert sha(path)==sha(ROOT/'src'/path.name)
    prov=json.loads((src/'provenance.json').read_text(encoding='utf-8'));assert prov['training_data_sha256']==sha(old/'data.pt') and prov['dense_checkpoint_sha256']==sha(old/'lr2/checkpoint.pt') and prov['old_result_sha256']==sha(old/'result.json');assert prov['plan_sha256']==sha(src/'plan.md')==sha(ROOT/'docs/router-author-sparse-plan-2026-09-14.md')
    d=torch.load(src/'evaluation-data.pt',map_location='cpu',weights_only=True);assert sha(src/'evaluation-data.pt')==prov['evaluation_data_sha256'];prior=torch.load(old/'data.pt',map_location='cpu',weights_only=True);validate(d['test']);validate(d['noisy'],False);newhashes=row_hashes(d['test']);assert len(newhashes)==1024
    for key in ['train','development','test']:assert not newhashes&row_hashes(prior[key])
    fill=torch.randint(0,8192,d['test']['inputs'].shape,generator=torch.Generator().manual_seed(2026091623));expected=d['test']['inputs'].clone();mask=expected==0;expected[mask]=fill[mask];assert torch.equal(expected,d['noisy']['inputs']) and torch.equal(d['test']['labels'],d['noisy']['labels'])
    for i,(q,s,t) in enumerate(d['swap_positions']):
        x=d['test']['inputs'][i];z=d['swapped']['inputs'][i];y=d['swapped']['labels'][i];assert int((x!=z).sum())==2 and torch.equal(x[32:],z[32:]);assert int((y!=-100).sum())==1 and int(y[q])==int(x[t])!=int(x[s]);assert x[s-1]==x[q]
    initial=torch.load(src/'initialization.pt',map_location='cpu',weights_only=True);oldinitial=torch.load(old/'initialization.pt',map_location='cpu',weights_only=True);assert weights_sha(initial['model'])==weights_sha(oldinitial)==prov['initial_backbone_sha'];assert weights_sha(initial['indexers'])==prov['initial_indexer_sha']
    cfg=json.loads((src/'config.json').read_text(encoding='utf-8'));assert cfg==author_configs()[1];assert [r['method'] for r in result['runs']]==list(METHODS);pre=json.loads((src/'preflight.json').read_text(encoding='utf-8'));assert pre['technical_main_updates']==32 and pre['technical_indexer_updates']==24 and pre['scientific_updates']==0
    rows=[];stamps=[];answers=0;development_answers=0
    dense_steps=2*19*100000*256*256
    target=d['test']['labels'][d['test']['labels']!=-100].reshape(1024,16).numpy();bank=d['test']['inputs'][:,1:32:2].numpy()
    all_runs=[result['dense_reference'],*result['runs']]
    for run in all_runs:
        method=run['method'];route=None
        if method=='dense_existing':
            ck=torch.load(old/'lr2/checkpoint.pt',map_location='cpu',weights_only=False);m=make_model(cfg,ck['model']);assert run['new_training_updates']==0 and run['original_updates']==7429
            original=json.loads((old/'lr2/result.json').read_text(encoding='utf-8'));dev=original['curves'][-1]['development_accuracy'];score=dict(main_score_entries=dense_steps,index_score_entries=0,teacher_distribution_entries=0);wall=original['wall_seconds'];train_seconds=original['train_step_seconds'];assert sha(old/'lr2/result.json')==run['original_result_sha256']
        else:
            folder=src/method;conf=json.loads((folder/'config.json').read_text(encoding='utf-8'));assert conf['config']==cfg and conf['initial_backbone_sha']==prov['initial_backbone_sha'];learned=method!='fixed_values6';assert conf['indexer_parameters']==(8256 if learned else 0) and conf['main_parameters']==1214720
            if learned:assert conf['initial_indexer_sha']==prov['initial_indexer_sha']
            events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()];steps=[e for e in events if e['event']=='optimizer_step'];epochs=[e for e in events if e['event']=='epoch'];assert events[0]['event']=='start' and events[-1]['event']=='fit_complete';assert [e['utc'] for e in events]==sorted(e['utc'] for e in events);assert [s['global_step'] for s in steps]==list(range(1,7430));assert [e['epoch'] for e in epochs]==list(range(1,20));assert run['epochs']==19 and run['main_updates']==7429 and run['indexer_updates']==(7429 if learned else 0)
            for epoch in range(1,20):
                ss=[s for s in steps if s['epoch']==epoch];assert [s['batch_examples'] for s in ss]==[256]*390+[160]
                for s in ss:
                    assert math.isclose(s['main_lr'],.01*.5*(1+math.cos(math.pi*(epoch-1)/64)),abs_tol=1e-12);assert s['indexer_lr']==(.001 if learned else None) and s['indexer_update']==learned;assert s['input_tokens']==s['batch_examples']*256 and s['supervised_answers']==s['batch_examples']*16
                    warm=method=='warm4_selectedkl_r16' and epoch<=4;main=2*s['batch_examples']*256*(256 if warm else 8);assert s['main_score_entries']==main and s['index_score_entries']==(2*s['batch_examples']*256*256 if learned else 0);assert s['teacher_distribution_entries']==(main if method.startswith('warm') else 0)
                    assert all(math.isfinite(s[k]) for k in ['ce','aux','main_gradient_norm','indexer_gradient_norm','step_wall_seconds'])
            for key in ['input_tokens','supervised_answers']:assert sum(s[key] for s in steps)==run[key]
            for key in ['main_score_entries','index_score_entries','teacher_distribution_entries']:assert sum(s[key] for s in steps)==run['score_accounting'][key]
            assert math.isclose(sum(s['step_wall_seconds'] for s in steps),run['train_step_seconds'],rel_tol=1e-10)
            ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==18 and ck['steps']==7429 and ck['epoch_boundary'] and ck['scheduler']['last_epoch']==19;assert all(float(v['step'])==7429 for v in ck['main_optimizer']['state'].values())
            if learned:assert all(float(v['step'])==7429 for v in ck['indexer_optimizer']['state'].values())
            else:assert ck['indexer_optimizer'] is None
            m,ix,route=build(cfg,ck['model'],ck['indexers'],method,'cpu');route.epoch=19;route.collect_aux=False;assert weights_sha(m.state_dict())==run['final_backbone_sha'] and weights_sha(ix.state_dict())==run['final_indexer_sha'];assert m.lm_head.weight is m.backbone.embeddings.word_embeddings.weight
            dev=evaluate(m,prior['development'],'cpu')['accuracy'];development_answers+=48000;assert dev==epochs[-1]['development_accuracy'];assert run['development_gate']==(dev>=.99);score=run['score_accounting'];wall=run['wall_seconds'];train_seconds=run['train_step_seconds'];stamps.append(dict(method=method,start=events[0]['utc'],finish=events[-1]['utc']))
        for key,n in [('test',16384),('swapped',1024),('noisy',16384)]:
            actual=evaluate(m,d[key],'cpu');assert actual['predictions']==run[key]['predictions'] and actual['accuracy']==run[key]['accuracy'] and actual['answers']==n;answers+=n
        if route:route.restore()
        clean=np.array(run['test']['predictions']).reshape(1024,16);noisy=np.array(run['noisy']['predictions']).reshape(1024,16);candidate_hit=float((clean[:,:,None]==bank[:,None,:]).any(-1).mean());delta=(clean==target).mean(-1)-(noisy==target).mean(-1);rng=np.random.default_rng(2026091628);bootstrap=[]
        for _ in range(10):bootstrap.extend(delta[rng.integers(0,1024,(500,1024))].mean(-1).tolist())
        interval=np.quantile(bootstrap,[.025,.975]).tolist();gate=result['control_valid'] and dev>=.99 and all(run[k]['accuracy']>=.99 and run[k]['accuracy']>=result['dense_reference'][k]['accuracy']-.01 for k in ['test','swapped'])
        if method!='dense_existing':assert gate==run['gate']
        rows.append(dict(method=method,development=dev,test=run['test']['accuracy'],swapped=run['swapped']['accuracy'],noisy=run['noisy']['accuracy'],gate=gate,candidate_value_hit=candidate_hit,noise_accuracy_drop=float(delta.mean()),paired_sequence_bootstrap95_drop=interval,score_accounting=score,main_score_ratio_to_dense=score['main_score_entries']/dense_steps,train_seconds=train_seconds,wall_seconds=wall))
    assert result['control_valid']==all(result['dense_reference'][k]['accuracy']>=.99 for k in ['test','swapped']);assert result['main_updates']==22287 and result['indexer_updates']==14858 and result['input_tokens']==1459200000 and result['supervised_answers']==91200000
    summary=dict(utc=now(),status='audited',files_verified=len(manifest),all_gpu_predictions_reproduced=True,answers_replayed=answers,development_answers_rechecked=development_answers,optimizer_updates=0,control_valid=result['control_valid'],rows=rows,training_stamps=stamps,analysis_wall_seconds=time.perf_counter()-timer);save(out/'audit-and-summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.4));dense=json.loads((old/'lr2/result.json').read_text(encoding='utf-8'));curves=[('dense_existing',dense['curves']),*[(r['method'],r['curves']) for r in result['runs']]]
    labels={'dense_existing':'Dense reference','warm4_selectedkl_r16':'4-epoch warmup + selected KL','ksa_additive_r16':'KSA-style additive router','fixed_values6':'Fixed six source values'}
    for method,cs in curves:
        axes[0].plot([x['epoch'] for x in cs],[x['development_accuracy']*100 for x in cs],label=labels[method]);axes[1].plot([x['epoch'] for x in cs],[x['train_nll'] for x in cs],label=labels[method])
    axes[0].set_ylabel('Development accuracy (%)');axes[0].set_ylim(0,102);axes[0].axhline(99,color='gray',ls=':');axes[1].set_ylabel('Training NLL');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('Epoch (100,000 examples)');ax.grid(alpha=.2)
    fig.suptitle('Known sparse recipes | same initial backbone and 19-epoch budget | one seed');fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);fig.savefig(out/'curves.pdf');plt.close(fig)
    save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(summary))

if __name__=='__main__':main()
