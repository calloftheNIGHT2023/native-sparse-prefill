"""Audit resumed training and compare 40/80 epochs on the same NEW test, without training."""
import hashlib,json,math,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from resume_joint_token_router import restore_checkpoint
from run_frozen_router import now,save,evaluate,sha,weights_sha
from zoology_entry import UPSTREAM

def main():
    src=ROOT/'results/joint-token-router-epoch80-v0';r=json.loads((src/'result.json').read_text());assert r['status']=='complete'
    out=ROOT/'results/joint-token-epoch80-analysis-v0';out.mkdir(exist_ok=False);torch.set_num_threads(4);timer=time.perf_counter()
    shutil.copy2(__file__,out/'analysis-source.py')
    manifest=json.loads((src/'manifest.json').read_text())
    for row in manifest:assert sha(src/row['path'])==row['sha256'],row['path']
    recovery=json.loads((src/'recovery.json').read_text());assert r['recovered_terminal_logging_error'] and r['scientific_updates_repeated']==recovery['scientific_updates_repeated']==0
    assert not (src/'RECOVERY_FAILURE.json').exists() and (src/'SUCCESS.json').exists()
    assert sha(src/'FAILURE.json')==recovery['original_failure_sha256'] and sha(src/'new-test.pt')==recovery['new_test_sha256']
    assert len(recovery['recovered'])==1 and recovery['recovered'][0]['method']=='exact_r16_shadow'
    assert sha(src/'exact_r16_shadow/checkpoint.pt')==recovery['recovered'][0]['checkpoint_sha256']
    failure=json.loads((src/'exact_r16_shadow/FAILURE.json').read_text());assert failure['global_step']==25040 and "multiple values for keyword argument 'utc'" in failure['traceback']
    for name in ['resume_joint_token_router.py','recover_router_epoch80_batch.py']:assert sha(src/'recovery-source'/name)==sha(ROOT/'src'/name)
    for saved in (src/'source').glob('*.py'):
        if saved.name!='resume_joint_token_router.py':assert sha(saved)==sha(ROOT/'src'/saved.name),saved.name
    for saved in (src/'source/upstream').rglob('*'):
        if saved.is_file():assert sha(saved)==sha(UPSTREAM/saved.relative_to(src/'source/upstream'))
    data=torch.load(src/'new-test.pt',map_location='cpu',weights_only=True);rows=[];curves={};rechecked=0;old_evaluated=0;comparisons=[]
    labels=data['fresh']['labels'];target=labels[labels!=-100].numpy().reshape(1024,4);rng=np.random.default_rng(2026091482)
    for run in r['runs']:
        name=run['method'];folder=src/name;oldfolder=ROOT/'results/joint-token-router-v0'/name
        ev=[json.loads(t) for t in (folder/'events.jsonl').read_text().splitlines()];steps=[t for t in ev if t['event']=='optimizer_step'];epochs=[t for t in ev if t['event']=='epoch']
        assert [s['global_step'] for s in steps]==list(range(12521,25041)) and [s['epoch'] for s in epochs]==list(range(41,81))
        assert [e['utc'] for e in ev]==sorted(e['utc'] for e in ev)
        assert all(math.isfinite(s[k]) for s in steps for k in ['ce','kl_sum_layers','main_gradient_norm','indexer_gradient_norm'])
        assert sum(s['batch_examples'] for s in steps)==400000
        if name=='exact_r16_shadow':assert ev[-1]['event']=='failed' and ev[-1]['global_step']==25040
        else:assert ev[-1]['event']=='complete' and not (folder/'FAILURE.json').exists()
        lineage=json.loads((folder/'resume-lineage.json').read_text());assert lineage['parent_checkpoint_sha256']==sha(oldfolder/'checkpoint.pt')
        ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['steps']==25040 and ck['epoch']==79 and ck['epoch_boundary']
        assert ck['scheduler']['last_epoch']==80
        for key in ['main_optimizer','indexer_optimizer']:assert all(float(v['step'])==25040 for v in ck[key]['state'].values())
        m,ix,route,opt,iopt,sched=restore_checkpoint(ck,'cpu');m.eval();route.collect_aux=False
        for key in ['fresh','swapped']:
            actual=evaluate(m,data[key],'cpu');assert actual['predictions']==run[key]['predictions'],(name,key);rechecked+=actual['answers']
        assert weights_sha(m.state_dict())==run['final_backbone_hash'];route.restore()
        before=torch.load(oldfolder/'checkpoint.pt',map_location='cpu',weights_only=False);m,ix,route,opt,iopt,sched=restore_checkpoint(before,'cpu');m.eval();route.collect_aux=False
        oldfresh=evaluate(m,data['fresh'],'cpu');oldswap=evaluate(m,data['swapped'],'cpu');route.restore();old_evaluated+=oldfresh['answers']+oldswap['answers']
        delta=(np.asarray(run['fresh']['predictions']).reshape(1024,4)==target).mean(1)-(np.asarray(oldfresh['predictions']).reshape(1024,4)==target).mean(1)
        boot=delta[rng.integers(0,1024,(5000,1024))].mean(1);oldresult=json.loads((oldfolder/'result.json').read_text());allcurves=oldresult['curves']+epochs;curves[name]=allcurves
        first99=next((c['epoch'] for c in allcurves if c['development_accuracy']>=.99),None)
        row=dict(method=name,new_test_epoch40=oldfresh['accuracy'],new_test_epoch80=run['fresh']['accuracy'],swap_epoch40=oldswap['accuracy'],swap_epoch80=run['swapped']['accuracy'],first_development_99_epoch=first99,additional_wall_seconds=run['wall_seconds'],additional_main_updates=12520,additional_indexer_updates=12520,paired_test_gain=float(delta.mean()),sequence_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),epoch80_gate=run.get('epoch80_gate'),start_utc=ev[0]['utc'],finish_utc=ev[-1]['utc'],parent_sha256=lineage['parent_checkpoint_sha256'])
        rows.append(row);comparisons.append(dict(method=name,epoch40_fresh=oldfresh,epoch40_swapped=oldswap))
    summary=dict(utc=now(),status='audited',manifest_files_verified=len(manifest),new_gpu_predictions_rechecked_on_cpu=rechecked,old_checkpoints_evaluated_on_new_test_answers=old_evaluated,all_new_gpu_predictions_equal=True,additional_main_updates=37560,additional_indexer_updates=37560,additional_input_tokens=76800000,additional_supervised_answers=4800000,analysis_optimizer_updates=0,new_test_seed=2026091481,control_valid=r['control_valid'],rows=rows,scope='Single-seed conditional extension; comparisons use a common new test and bootstrap reflects test-row uncertainty only',analysis_wall_seconds=time.perf_counter()-timer)
    save(out/'epoch40-on-new-test.json',comparisons)
    summary.update(recovered_terminal_logging_error_verified=True,scientific_updates_repeated=0,recovery=recovery,batch_wall_seconds=r['wall_seconds'],fit_wall_seconds_sum=r['fit_wall_seconds_sum'])
    save(out/'audit-and-summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for name,cs in curves.items():
        ax[0].plot([c['epoch'] for c in cs],[c['development_accuracy']*100 for c in cs],label=name)
        ax[1].plot([c['epoch'] for c in cs],[c['train_nll'] for c in cs],label=name)
    for a in ax:a.axvline(40,color='gray',linestyle='--',alpha=.7);a.grid(alpha=.2);a.set_xlabel('Total epochs')
    ax[0].set_ylabel('Development accuracy (%)');ax[0].axhline(99,color='gray',linestyle=':',alpha=.4);ax[0].legend(fontsize=8)
    ax[1].set_ylabel('Training NLL');fig.suptitle('40 to 80 epoch continuation | migrated A40 | same single seed | dense KL teacher');fig.tight_layout();fig.savefig(out/'continuation-curves.png',dpi=160);fig.savefig(out/'continuation-curves.pdf');plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
