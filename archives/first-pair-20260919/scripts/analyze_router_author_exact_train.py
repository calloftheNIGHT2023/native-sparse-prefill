"""Audit the additional exact-native fit and re-evaluate all five frozen checkpoints on CPU."""
import json,math,shutil,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from router_author_control import author_configs,make_model,validate,row_hashes,evaluate
from router_author_sparse import build
from run_router_author_exact_train import exact_model
from run_frozen_router import now,save,sha,weights_sha
from zoology_sparse_schedule import retained_edges

def main():
    torch.set_num_threads(4);timer=time.perf_counter();src=ROOT/'results/router-author-exact-train-v0';old=ROOT/'results/router-author-control-v0';prior=ROOT/'results/router-author-sparse-v0';out=ROOT/'results/router-author-exact-analysis-v0';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py');r=json.loads((src/'result.json').read_text(encoding='utf-8'));assert (src/'SUCCESS.json').exists() and not list(src.rglob('FAILURE.json'));manifest=json.loads((src/'manifest.json').read_text(encoding='utf-8'))
    for row in manifest:assert sha(src/row['path'])==row['sha256']
    if r.get('recovered_metadata_only'):
        recovery=json.loads((src/'recovery.json').read_text(encoding='utf-8'));assert recovery['optimizer_updates']==0 and (src/'RECOVERED_FAILURE.json').exists()
        for name,h in recovery['protected_files_unchanged'].items():assert sha(src/name)==h
        assert sha(src/'source/recover_router_author_exact_metadata.py')==recovery['source_sha256']
    for path in (src/'source').glob('*.py'):assert sha(path)==sha(ROOT/'src'/path.name)
    p=json.loads((src/'provenance.json').read_text(encoding='utf-8'));assert sha(old/'data.pt')==p['training_data_sha256'] and sha(prior/'result.json')==p['previous_batch_result_sha256'];assert sha(src/'plan.md')==p['plan_sha256']==sha(ROOT/'docs/router-author-exact-train-plan-2026-09-14.md')
    initial=torch.load(src/'initialization.pt',weights_only=True,map_location='cpu');assert weights_sha(initial)==p['initial_backbone_sha']==weights_sha(torch.load(old/'initialization.pt',weights_only=True,map_location='cpu'));cfg=json.loads((src/'config.json').read_text(encoding='utf-8'));assert cfg==author_configs()[1]
    data=torch.load(src/'evaluation-data.pt',map_location='cpu',weights_only=True);audit=json.loads((src/'data-audit.json').read_text(encoding='utf-8'));assert sha(src/'evaluation-data.pt')==audit['evaluation_data_sha256'];validate(data['test']);validate(data['noisy'],False);hh=row_hashes(data['test']);assert len(hh)==1024
    olddata=torch.load(old/'data.pt',weights_only=True,map_location='cpu');previous=torch.load(prior/'evaluation-data.pt',weights_only=True,map_location='cpu')
    for key in ['train','development','test']:assert not hh&row_hashes(olddata[key])
    assert not hh&row_hashes(previous['test']);fill=torch.randint(0,8192,data['test']['inputs'].shape,generator=torch.Generator().manual_seed(2026091643));expected=data['test']['inputs'].clone();mask=expected==0;expected[mask]=fill[mask];assert torch.equal(expected,data['noisy']['inputs']) and torch.equal(data['test']['labels'],data['noisy']['labels'])
    for i,(q,s,t) in enumerate(data['swap_positions']):
        x=data['test']['inputs'][i];z=data['swapped']['inputs'][i];y=data['swapped']['labels'][i];assert int((x!=z).sum())==2 and torch.equal(x[32:],z[32:]);assert int((y!=-100).sum())==1 and int(y[q])==int(x[t])!=int(x[s]) and x[s-1]==x[q]
    events=[json.loads(line) for line in (src/'events.jsonl').read_text(encoding='utf-8').splitlines()];steps=[s for s in events if s['event']=='optimizer_step'];epochs=[e for e in events if e['event']=='epoch'];assert events[0]['event']=='start' and events[-1]['event']=='fit_complete';assert [e['utc'] for e in events]==sorted(e['utc'] for e in events);assert [s['global_step'] for s in steps]==list(range(1,7430));assert [e['epoch'] for e in epochs]==list(range(1,20))
    for epoch in range(1,20):
        ss=[s for s in steps if s['epoch']==epoch];assert [s['batch_examples'] for s in ss]==[256]*390+[160]
        for s in ss:
            assert math.isclose(s['main_lr'],.01*.5*(1+math.cos(math.pi*(epoch-1)/64)),abs_tol=1e-12);assert s['input_tokens']==s['batch_examples']*256 and s['supervised_answers']==s['batch_examples']*16;assert s['full_main_score_entries']==2*s['batch_examples']*256*256 and s['selected_attention_edges']==2*s['batch_examples']*retained_edges(256,8) and s['indexer_updates']==0;assert all(math.isfinite(s[k]) for k in ['ce','main_gradient_norm','step_wall_seconds'])
    assert r['main_updates']==7429 and r['indexer_updates']==0 and r['technical_optimizer_updates']==0;assert sum(s['input_tokens'] for s in steps)==r['input_tokens']==486400000 and sum(s['supervised_answers'] for s in steps)==r['supervised_answers']==30400000;assert math.isclose(sum(s['step_wall_seconds'] for s in steps),r['fit']['train_step_seconds'],rel_tol=1e-10)
    ck=torch.load(src/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==18 and ck['steps']==7429 and ck['epoch_boundary'] and ck['scheduler']['last_epoch']==19;assert all(float(v['step'])==7429 for v in ck['optimizer']['state'].values());assert weights_sha(ck['model'])==r['fit']['final_backbone_sha']
    refs=json.loads((src/'reference-checkpoints.json').read_text(encoding='utf-8'));assert [ref['method'] for ref in refs]==[row['method'] for row in r['conditions']];rows=[];count=0;dev_count=0
    for ref,row in zip(refs,r['conditions']):
        path=ROOT/ref['path'];assert sha(path)==ref['sha256'];c=torch.load(path,map_location='cpu',weights_only=False);route=None;method=row['method']
        if method=='exact_native':m=exact_model(cfg,c['model'],'cpu')
        elif method=='dense_existing':m=make_model(cfg,c['model'])
        else:m,ix,route=build(cfg,c['model'],c['indexers'],method,'cpu');route.epoch=19;route.collect_aux=False
        assert weights_sha(m.state_dict())==ref['model_sha256'] and m.lm_head.weight is m.backbone.embeddings.word_embeddings.weight
        for key,n in [('test',16384),('swapped',1024),('noisy',16384)]:
            got=evaluate(m,data[key],'cpu');assert got['predictions']==row[key]['predictions'] and got['accuracy']==row[key]['accuracy'] and got['answers']==n;count+=n
        if method=='exact_native':assert evaluate(m,olddata['development'],'cpu')['accuracy']==row['development_accuracy']==epochs[-1]['development_accuracy'];dev_count+=48000
        if route:route.restore()
        assert weights_sha(m.state_dict())==ref['model_sha256'];assert row['new_training_updates']==(7429 if method=='exact_native' else 0);gate=r['control_valid'] and row['development_accuracy']>=.99 and all(row[k]['accuracy']>=.99 and row[k]['accuracy']>=r['conditions'][0][k]['accuracy']-.01 for k in ['test','swapped']);assert gate==row['gate'];rows.append(dict(method=method,development=row['development_accuracy'],test=row['test']['accuracy'],swapped=row['swapped']['accuracy'],noisy=row['noisy']['accuracy'],gate=gate))
    assert r['control_valid']==all(r['conditions'][0][k]['accuracy']>=.99 for k in ['test','swapped']);summary=dict(utc=now(),status='audited',files_verified=len(manifest),all_gpu_predictions_reproduced=True,answers_replayed=count,development_answers_rechecked=dev_count,optimizer_updates=0,rows=rows,control_valid=r['control_valid'],fit_start=events[0]['utc'],fit_finish=events[-1]['utc'],analysis_wall_seconds=time.perf_counter()-timer);save(out/'audit-and-summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dense=json.loads((old/'lr2/result.json').read_text(encoding='utf-8'));sparse=json.loads((prior/'result.json').read_text(encoding='utf-8'));labels={'dense_existing':'Dense reference','exact_native':'Exact-QK top8 from step 1','warm4_selectedkl_r16':'Warmup + selected KL','ksa_additive_r16':'KSA-style router','fixed_values6':'Fixed six values'};fig,ax=plt.subplots(figsize=(8.8,5))
    for method,cs in [('dense_existing',dense['curves']),*[(v['method'],v['curves']) for v in sparse['runs']],('exact_native',r['fit']['curves'])]:ax.plot([v['epoch'] for v in cs],[v['development_accuracy']*100 for v in cs],label=labels[method])
    ax.set(xlabel='Epoch (100,000 examples)',ylabel='Development accuracy (%)',ylim=(0,102),title='One initialization, same main optimizer and 19-epoch budget');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);fig.savefig(out/'curves.pdf');plt.close(fig);save(out/'manifest.json',[dict(path=path.relative_to(out).as_posix(),sha256=sha(path)) for path in sorted(out.rglob('*')) if path.is_file()]);print(json.dumps(summary))

if __name__=='__main__':main()
