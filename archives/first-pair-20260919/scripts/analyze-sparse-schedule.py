"""Read-only, full-run audit and paired fresh/intervention analysis of frozen screen."""
import argparse,hashlib,json,math,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from zoology_entry import LanguageModel
from zoology.config import ModelConfig
from zoology_sparse_schedule import install,set_epoch,SCHEDULES,retained_edges

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main(args):
    run=args.run;out=args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    manifest=read(run/'manifest.json')
    for item in manifest:assert sha(run/item['path'])==item['sha256'],item['path']
    final=read(run/'result.json');assert final['status']=='complete'
    common=torch.load(run/'shared-data-and-initialization.pt',map_location='cpu',weights_only=True)
    data_hashes={}
    for name,split in common['splits'].items():
        hashes=[hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in split['inputs']]
        assert len(hashes)==len(set(hashes));data_hashes[name]=set(hashes)
        for xx,yy in zip(split['inputs'],split['labels']):
            mapping={int(xx[i]):int(xx[i+1]) for i in range(0,8,2)}
            assert len(mapping)==4 and len(set(mapping.values()))==4 and int((yy!=-100).sum())==4
            for pos in torch.where(yy!=-100)[0]:assert int(pos)>=8 and mapping[int(xx[pos])]==int(yy[pos])
    assert not data_hashes['train']&data_hashes['validation'] and not data_hashes['fresh']&(data_hashes['train']|data_hashes['validation'])
    fresh=common['splits']['fresh'];x=fresh['inputs'];y=fresh['labels'];length=x.shape[1]
    labels=y[y!=-100].numpy();qpos=[];source=[];alternate=[];old=[];new=[]
    changed_query=x.clone();changed_values=x.clone()
    for i,(row,lab) in enumerate(zip(x,y)):
        q=int(torch.where(lab!=-100)[0][-1]);s=next(p+1 for p in range(0,8,2) if int(row[p])==int(row[q]));a=((s-1+2)%8)+1
        qpos.append(q);source.append(s);alternate.append(a);old.append(int(row[s]));new.append(int(row[a]))
        changed_query[i,q]=row[a-1];changed_values[i,s]=row[a];changed_values[i,a]=row[s]
        assert old[-1]!=new[-1] and int((changed_values[i]!=row).sum())==2
        assert torch.equal(changed_values[i,8:],row[8:]) and int((changed_query[i]!=row).sum())==1
    torch.save(dict(original=x,changed_query=changed_query,changed_values=changed_values,query_positions=qpos,
        source_value_positions=source,old_labels=old,new_labels=new),out/'intervention-inputs.pt')
    methods=[v['method'] for v in final['runs']];rows=[];scores={};rng=np.random.default_rng(2026091441)
    bootstrap=rng.integers(0,len(x),(4000,len(x)))
    for method in methods:
        folder=run/method;r=read(folder/'result.json');cfg=read(folder/'config.json');events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
        steps=[v for v in events if v['event']=='optimizer_step'];vs=[v for v in events if v['event']=='validation']
        assert len(steps)==r['updates']==313*r['epochs'] and len(vs)==r['epochs']
        assert [v['step'] for v in steps]==list(range(1,len(steps)+1))
        assert all(a['utc']<=b['utc'] for a,b in zip(events,events[1:]))
        assert all(math.isfinite(v['loss']) and math.isfinite(v['gradient_norm']) for v in steps)
        assert steps[-1]['input_tokens']==r['input_tokens']==r['epochs']*10000*length
        cumulative=0
        for step in steps:
            cumulative+=step['batch_examples']*sum(retained_edges(length,length if step['epoch']<SCHEDULES[method][i] else 8) for i in range(2))
            assert cumulative==step['cumulative_retained_edges']
            assert math.isclose(step['learning_rate'],.001*(1+math.cos(math.pi*step['epoch']/100))/2,rel_tol=1e-10)
        assert cumulative==r['cumulative_retained_edges']
        p=np.asarray(r['fresh']['predictions']);correct=(p==labels);assert correct.mean()==r['fresh']['accuracy']
        scores[method]=correct.reshape(len(x),4).mean(1)
        ci=np.quantile(scores[method][bootstrap].mean(1),[.025,.975]).tolist()
        checkpoint=folder/'checkpoint.pt';digest=sha(checkpoint);state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['steps']==r['updates']
        model=LanguageModel(ModelConfig(**state['config']['model']));model.load_state_dict(state['model']);install(model,method);set_epoch(model,r['epochs']);model.eval()
        def predict(inputs,coverage=False):
            pred=[];covered=[[],[]]
            with torch.no_grad():
                for start in range(0,len(inputs),32):
                    logits=model(inputs[start:start+32]);qs=torch.tensor(qpos[start:start+32]);ss=torch.tensor(source[start:start+32]);bs=torch.arange(len(qs))
                    pred.extend(logits[bs,qs].argmax(-1).tolist())
                    if coverage:
                        for i,layer in enumerate(model.backbone.layers):covered[i].extend(layer.sequence_mixer.inner_attn.last_mask[bs,0,qs,ss].tolist())
            return np.asarray(pred),covered
        before,coverage=predict(x,True);query,_=predict(changed_query);values,_=predict(changed_values)
        orig=np.asarray(old);target=np.asarray(new);gap=np.asarray(qpos)-np.asarray(source);far=gap>length//2
        assert sha(checkpoint)==digest
        probe=dict(original_last_accuracy=float((before==orig).mean()),changed_query_accuracy=float((query==target).mean()),
            swapped_values_accuracy=float((values==target).mean()),query_changed_fraction=float((query!=before).mean()),
            source_values_changed_fraction=float((values!=before).mean()),far_examples=int(far.sum()),
            far_swapped_values_accuracy=float((values[far]==target[far]).mean()) if far.any() else None,
            last_query_source_value_selected_fraction=[float(np.mean(c)) for c in coverage],
            cpu_original_prediction_agreement_with_saved_gpu=float((before==p.reshape(len(x),4)[:,-1]).mean()),
            optimizer_updates=0,checkpoint_sha256=digest,
            scope='Post-training diagnostics on same fresh examples. Query-key mutation can be OOD; source swap preserves unique mapping. Selection coverage is not a causal information-flow bound.')
        save(out/(method+'-probe.json'),probe)
        save(out/(method+'-probe-predictions.json'),dict(original=before.tolist(),query=query.tolist(),values=values.tolist(),old=old,new=new))
        milestones={str(t):next((dict(epoch=v['epoch'],step=v['step'],edges=v['logical_retained_edges']) for v in vs if v['validation_accuracy']>=t),None) for t in [.9,.99]}
        row=dict(method=method,accuracy=float(correct.mean()),sequence_bootstrap95=ci,validation_accuracy=vs[-1]['validation_accuracy'],
            milestones=milestones,wall_seconds=r['wall_seconds'],training_seconds=r['training_seconds'],peak_allocated_bytes=r['peak_allocated_bytes'],
            cumulative_retained_edges=cumulative,updates=r['updates'],input_tokens=r['input_tokens'],probe=probe)
        rows.append(row)
    assert len({read(run/m/'result.json')['initial_hash'] for m in methods})==1
    contrasts=[];gate=[];lookup={r['method']:r for r in rows}
    for candidate in ['first_warm8','second_warm8']:
        if candidate not in lookup:continue
        accuracy_pass=True;edge_pass=True
        for baseline in ['native','all_warm4']:
            if baseline not in lookup:accuracy_pass=False;edge_pass=False;continue
            delta=scores[candidate]-scores[baseline]
            contrasts.append(dict(candidate=candidate,baseline=baseline,delta_accuracy=float(delta.mean()),
                paired_sequence_bootstrap95=np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist()))
            accuracy_pass &= delta.mean()>=.05
            ct=lookup[candidate]['milestones']['0.99'];bt=lookup[baseline]['milestones']['0.99']
            edge_pass &= ct is not None and bt is not None and ct['edges']<=.8*bt['edges'] and delta.mean()>=-.01
        gate.append(dict(candidate=candidate,accuracy_gate=bool(accuracy_pass),retained_edge_gate=bool(edge_pass),
            pass_to_independent_confirmation=bool(accuracy_pass or edge_pass)))
    if all(m in lookup for m in ['all_warm4','first_warm8','second_warm8']):assert len({lookup[m]['cumulative_retained_edges'] for m in ['all_warm4','first_warm8','second_warm8']})==1
    save(out/'analysis.json',dict(utc=datetime.now(timezone.utc).isoformat(),manifest_files_verified=len(manifest),data_labels_and_split_disjointness_checked=True,runs=rows,
        contrasts=contrasts,frozen_screen_gates=gate,training_seed_repeats=1,additional_optimizer_updates=0,
        scope='Exploratory one-seed screen. Fresh sequence confidence intervals do not measure seed-to-seed uncertainty. Retained-edge counts are not FLOPs or actual speedups.'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for method in methods:
        curves=read(run/method/'result.json')['curves'];acc=[100*v['validation_accuracy'] for v in curves]
        axes[0].plot([v['epoch'] for v in curves],acc,label=method)
        axes[1].plot([v['logical_retained_edges']/1e6 for v in curves],acc,label=method)
    for ax in axes:ax.set(ylabel='Validation accuracy (%)',ylim=(0,101));ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_xlabel('Epoch');axes[1].set_xlabel('Cumulative retained edges (million; not FLOPs)')
    fig.tight_layout();fig.savefig(out/'schedule-curves.png',dpi=180);plt.close(fig)
    shutil.copy2(__file__,out/Path(__file__).name)
    save(out/'manifest.json',[dict(path=p.name,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()])
    print(json.dumps(dict(rows=[{k:r[k] for k in ['method','accuracy','milestones','wall_seconds']} for r in rows],gate=gate)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
