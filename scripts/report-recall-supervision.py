"""Audit paired inputs and diagnose query use at fixed final checkpoints."""
import hashlib,json,math,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_recall_learnability import dataset,prediction,evaluate
from recall_supervision import add_training_queries
from far_recall import symbolic_lookup
from recall_binding import install_binding_control

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def ci_family(preds,num_values,seed=2026091411):
    a=np.asarray(preds).reshape(-1,num_values);scores=(a==np.arange(num_values)).mean(1)
    rng=np.random.default_rng(seed);means=scores[rng.integers(0,len(scores),(4000,len(scores)))].mean(1)
    return dict(accuracy=float(scores.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),unit='counterfactual family',scope='Evaluation uncertainty conditional on this seed; not training-seed uncertainty')

@torch.no_grad()
def query_probe(folder,cfg):
    ck=folder/'checkpoint.pt';digest=sha(ck)
    c=GPTNeoXConfig.from_dict(json.loads((folder/'model-config.json').read_text()));c._attn_implementation='eager'
    model=GPTNeoXForCausalLM(c).float();model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model']);model.eval()
    install_binding_control(model,cfg)
    x,y,metas=dataset(cfg,'validation');sweep=[];labels=[];train_format=[]
    for fi,meta in enumerate(metas):
        block=x[fi*cfg['num_values']:(fi+1)*cfg['num_values']]
        train_format.append(add_training_queries(block,meta,cfg)[0])
        for row in block:
            for p in sorted(meta['record_positions']):
                changed=row.clone();changed[-1]=row[p+1];sweep.append(changed);labels.append(int(row[p+2])-cfg['value_base'])
    xx=torch.stack(sweep);yy=torch.tensor(labels);assert torch.equal(symbolic_lookup(xx,cfg),yy)
    predicted=[]
    for start in range(0,len(xx),cfg['batch_size']):
        lg=prediction(model,xx[start:start+cfg['batch_size']])[:,cfg['value_base']:cfg['value_base']+cfg['num_values']]
        predicted.extend(lg.argmax(-1).tolist())
    pp=torch.tensor(predicted).reshape(-1,cfg['records']);truth=yy.reshape_as(pp)
    flips=[];both=[]
    for p,t in zip(pp,truth):
        for a in range(len(p)):
            for b in range(a+1,len(p)):
                if t[a]!=t[b]:flips.append(bool(p[a]!=p[b]));both.append(bool(p[a]==t[a] and p[b]==t[b]))
    native=evaluate(model,(torch.cat(train_format),y,metas),cfg)
    assert sha(ck)==digest
    return dict(started_on='development only',families=len(metas),query_examples=len(xx),query_sweep_accuracy=float((pp==truth).float().mean()),
        distinct_answer_pairs=len(flips),prediction_changed_fraction=sum(flips)/len(flips),both_correct_fraction=sum(both)/len(both),
        multiquery_input_last_answer_accuracy=native['accuracy'],checkpoint_sha256=digest,optimizer_updates=0,
        predictions=predicted,labels=labels,scope='Correlated counterfactual/query groups; exploratory diagnostic, not independent trials')

def main():
    started=now();torch.set_num_threads(4);out=ROOT/'results/recall-supervision-pair-v0';out.mkdir(exist_ok=False)
    names=['recall-supervision-last-v0','recall-supervision-all-v0'];runs=[];steps=[];configs=[];checks=[]
    for name in names:
        p=ROOT/'results'/name;r=json.loads((p/'result.json').read_text());assert r['status']=='complete'
        manifest=json.loads((p/'manifest.json').read_text())
        for x in manifest:assert sha(p/x['path'])==x['sha256']
        ev=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()];s=[x for x in ev if x['event']=='optimizer_step']
        cfg=json.loads((p/'frozen-config.json').read_text());configs.append(cfg)
        assert [x['step'] for x in s]==list(range(1,4001));assert all(a['utc']<=b['utc'] for a,b in zip(ev,ev[1:]))
        assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) and x['grad_norm']>0 for x in s)
        hashes=[h for x in s for h in x['fresh_family_hashes']];assert len(hashes)==len(set(hashes))==64000
        heldout={m['family_sha256'] for records in json.loads((p/'data-families.json').read_text()).values() for m in records}
        assert not heldout.intersection(hashes)
        assert s[-1]['supervised_answers']==r['supervised_answer_tokens']
        r['name']=name;r['test_family_bootstrap']=ci_family(r['dense_test']['predictions'],cfg['num_values'])
        qp=query_probe(p,cfg);save(out/(name+'-query-probe.json'),qp)
        r['query_probe']={k:v for k,v in qp.items() if k not in ['predictions','labels']}
        runs.append(r);steps.append(s);checks.append(dict(name=name,manifest_files_verified=len(manifest),ordered_updates=len(s),fresh_families=64000,holdout_overlap=0))
    assert runs[0]['initial_hash']==runs[1]['initial_hash']
    diffs=[k for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k)]
    assert set(diffs)=={'name','supervision'}
    for a,b in zip(*steps):
        for k in ['step','lr','fresh_family_hashes','fresh_family_variants','input_tokens']:assert a[k]==b[k],(a['step'],k)
    # Same source implementation + family hashes + variants + query positions imply identical inputs.
    for name in ['far_recall.py','recall_supervision.py','run_recall_learnability.py']:
        assert sha(ROOT/'results'/names[0]/'source-snapshot'/name)==sha(ROOT/'results'/names[1]/'source-snapshot'/name)
    paired=dict(initialization_equal=True,all_4000_input_batches_identical=True,learning_rate_schedule_identical=True,
        differing_config_keys=diffs,supervised_answer_tokens=[r['supervised_answer_tokens'] for r in runs],checks=checks)
    result=dict(started_utc=started,finished_utc=now(),runs=runs,paired_audit=paired,gpu_jobs_started=0,cloud_billing_status='not checked')
    save(out/'result.json',result);save(ROOT/'logs/recall-supervision-pair-audit.json',paired)
    fig,ax=plt.subplots(figsize=(8,4))
    for r,label in zip(runs,['Last query supervised','All four queries supervised']):
        ax.plot([d['step'] for d in r['development']],[100*d['accuracy'] for d in r['development']],marker='o',label=label)
    ax.axhline(80,color='grey',ls='--',label='Fixed 80% gate');ax.set(xlabel='Optimizer updates',ylabel='Single-query development accuracy (%)',ylim=(0,105));ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'development.png',dpi=170);plt.close(fig)
    snapshot=out/'source-snapshot';snapshot.mkdir();shutil.copy2(__file__,snapshot/Path(__file__).name)
    save(out/'manifest.json',[dict(path=str(p.relative_to(out)),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()])
    print(json.dumps(dict(paired=paired,results=[dict(name=r['name'],dev=r['development'][-1]['accuracy'],test=r['dense_test']['accuracy'],gate=r['learnability_gate_passed'],query=r['query_probe']) for r in runs])))

if __name__=='__main__':main()
