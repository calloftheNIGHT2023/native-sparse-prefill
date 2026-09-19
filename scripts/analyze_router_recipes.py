"""Independent CPU prediction replay, source/lineage audit and matched-budget report."""
import json,math,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_router_known_recipes import build
from resume_joint_token_router import restore_checkpoint
from run_frozen_router import now,save,sha,weights_sha,evaluate

def main():
    torch.set_num_threads(4);src=ROOT/'results/router-known-recipes-v0';out=ROOT/'results/router-known-recipes-analysis-v0';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py');timer=time.perf_counter()
    r=json.loads((src/'result.json').read_text());assert r['status']=='complete' and (src/'SUCCESS.json').exists() and not (src/'FAILURE.json').exists()
    manifest=json.loads((src/'manifest.json').read_text())
    for row in manifest:assert sha(src/row['path'])==row['sha256']
    for path in (src/'source').glob('*.py'):assert sha(path)==sha(ROOT/'src'/path.name)
    source=ROOT/'results/joint-token-router-v0';data=torch.load(src/'new-test.pt',map_location='cpu',weights_only=True);old=torch.load(source/'data-and-initialization.pt',map_location='cpu',weights_only=True);original_indexers=torch.load(source/'learned_r16/initial-indexer.pt',map_location='cpu',weights_only=True)
    assert sha(source/'learned_r16/checkpoint.pt')==r['reference']['checkpoint_sha256'];counts=0;rows=[];stamp=[];curves={};old_result=json.loads((source/'learned_r16/result.json').read_text())
    c=torch.load(source/'learned_r16/checkpoint.pt',map_location='cpu',weights_only=False);m,ix,route,*_=restore_checkpoint(c,'cpu');m.eval();route.collect_aux=False
    for key in ['fresh','swapped']:
        actual=evaluate(m,data[key],'cpu');assert actual['predictions']==r['reference'][key]['predictions'];counts+=actual['answers']
    route.restore();curves['fullkl_r16_epoch40_reference']=old_result['curves'];rows.append(dict(recipe='fullkl_r16_epoch40_reference',fresh=r['reference']['fresh']['accuracy'],swapped=r['reference']['swapped']['accuracy'],new_training_updates=0,first_dev99=None))
    for run in r['runs']:
        name=run['recipe'];folder=src/name;ev=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()];steps=[s for s in ev if s['event']=='optimizer_step'];epochs=[s for s in ev if s['event']=='epoch'];conf=json.loads((folder/'config.json').read_text())
        assert conf['initial_backbone_sha']==weights_sha(old['initial_state']) and conf['initial_indexer_sha']==weights_sha(original_indexers)
        assert [s['global_step'] for s in steps]==list(range(1,12521)) and [s['epoch'] for s in epochs]==list(range(1,41));assert sum(s['batch_examples'] for s in steps)==400000
        assert [e['utc'] for e in ev]==sorted(e['utc'] for e in ev) and ev[-1]['event']=='complete'
        assert all(math.isfinite(s[k]) for s in steps for k in ['ce','aux','main_gradient_norm','indexer_gradient_norm'])
        expected_main=10000*2*(4*64*64+36*64*8) if name=='warm4_selectedkl_r16' else 400000*2*64*8
        assert run['score_accounting']==dict(main_score_entries=expected_main,index_score_entries=400000*2*64*64,teacher_distribution_entries=expected_main if name=='warm4_selectedkl_r16' else 0)
        for k,v in run['score_accounting'].items():assert sum(s[k] for s in steps)==v
        c=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert c['epoch']==39 and c['steps']==12520 and c['epoch_boundary'] and c['scheduler']['last_epoch']==40
        for opt in ['main_optimizer','indexer_optimizer']:assert all(float(v['step'])==12520 for v in c[opt]['state'].values())
        m,ix,route=build(c,'cpu')
        for key in ['fresh','swapped']:
            actual=evaluate(m,data[key],'cpu');assert actual['predictions']==run[key]['predictions'],(name,key);counts+=actual['answers']
        route.restore();assert weights_sha(m.state_dict())==run['final_backbone_sha'] and weights_sha(ix.state_dict())==run['final_indexer_sha']
        first=next((c['epoch'] for c in epochs if c['development_accuracy']>=.99),None);curves[name]=epochs;rows.append(dict(recipe=name,fresh=run['fresh']['accuracy'],swapped=run['swapped']['accuracy'],first_dev99=first,new_main_updates=12520,new_indexer_updates=12520,wall_seconds=run['wall_seconds'],gate=run['gate'],score_accounting=run['score_accounting']));stamp.append(dict(recipe=name,start=ev[0]['utc'],finish=ev[-1]['utc']))
    summary=dict(status='audited',utc=now(),files_verified=len(manifest),all_cpu_predictions_match=True,answers_replayed=counts,analysis_optimizer_updates=0,additional_main_updates=25040,additional_indexer_updates=25040,additional_input_tokens=51200000,additional_supervised_answers=3200000,rows=rows,training_stamps=stamp,analysis_wall_seconds=time.perf_counter()-timer,scope='One initial seed and fixed 40-epoch budget; known recipe adaptations, not novel methods; no same-device throughput comparison to old reference')
    save(out/'audit-and-summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4.3))
    for name,cs in curves.items():
        ax[0].plot([c['epoch'] for c in cs],[c['development_accuracy']*100 for c in cs],label=name);ax[1].plot([c['epoch'] for c in cs],[c['train_nll'] for c in cs],label=name)
    ax[0].set_ylim(0,102);ax[0].set_ylabel('Development accuracy (%)');ax[0].axhline(99,color='gray',ls=':',alpha=.5);ax[0].legend(fontsize=7);ax[1].set_ylabel('Training NLL')
    for a in ax:a.set_xlabel('Epoch');a.grid(alpha=.2)
    fig.suptitle('Known recipe adaptations | same initial weights and 40 epochs | MQAR toy task');fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);fig.savefig(out/'curves.pdf');plt.close(fig);print(json.dumps(summary))

if __name__=='__main__':main()
