"""Audit this bounded local follow-up and evaluate new families without training."""
import json,hashlib,shutil,sys,importlib.util,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_recall_learnability import dataset,evaluate
from recall_binding import install_binding_control

spec=importlib.util.spec_from_file_location('paired_report',ROOT/'scripts/report-recall-supervision.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
NAMES=['recall-supervision-last-v0','recall-supervision-all-v0','recall-binding-predecessor-v0','recall-learning-rate-1e3-v0']
LABELS=['Last query, 3e-4','Four queries, 3e-4','Predecessor first layer, 3e-4','Last query, 1e-3']
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    started=now();torch.set_num_threads(4);out=ROOT/'results/recall-followup-v0';out.mkdir(exist_ok=False)
    runs=[];configs=[];allsteps=[];audits=[];allheldout=set()
    for name in NAMES:
        p=ROOT/'results'/name;r=json.loads((p/'result.json').read_text());assert r['status']=='complete'
        cfg=json.loads((p/'frozen-config.json').read_text());configs.append(cfg)
        manifest=json.loads((p/'manifest.json').read_text())
        for f in manifest:assert sha(p/f['path'])==f['sha256'],f['path']
        events=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()];steps=[x for x in events if x['event']=='optimizer_step']
        assert [x['step'] for x in steps]==list(range(1,4001))
        assert all(a['utc']<=b['utc'] for a,b in zip(events,events[1:]))
        assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) and x['grad_norm']>0 for x in steps)
        hashes=[h for x in steps for h in x['fresh_family_hashes']];assert len(set(hashes))==len(hashes)==64000
        heldout={m['family_sha256'] for ms in json.loads((p/'data-families.json').read_text()).values() for m in ms};assert not set(hashes)&heldout;allheldout.update(heldout)
        allsteps.append(steps);r['name']=name;r['test_ci']=helper.ci_family(r['dense_test']['predictions'],cfg['num_values'])
        audits.append(dict(name=name,manifest_files_verified=len(manifest),updates=len(steps),fresh_training_families=len(hashes)))
        runs.append(r)
    for i in range(1,len(runs)):
        assert runs[0]['initial_hash']==runs[i]['initial_hash']
        for a,b in zip(allsteps[0],allsteps[i]):
            for k in ['step','fresh_family_hashes','fresh_family_variants','input_tokens']:assert a[k]==b[k]
        assert sha(ROOT/'results'/NAMES[0]/'source-snapshot/far_recall.py')==sha(ROOT/'results'/NAMES[i]/'source-snapshot/far_recall.py')
        assert sha(ROOT/'results'/NAMES[0]/'source-snapshot/recall_supervision.py')==sha(ROOT/'results'/NAMES[i]/'source-snapshot/recall_supervision.py')
        diffs={k for k in set(configs[0])|set(configs[i]) if configs[0].get(k)!=configs[i].get(k)}
        allowed=[set(),{'name','supervision'},{'name','binding_control','scope'},{'name','learning_rate','scope'}][i]
        assert diffs==allowed,(i,diffs)
        if i!=3:assert [s['lr'] for s in allsteps[i]]==[s['lr'] for s in allsteps[0]]
    fresh_cfg=dict(configs[0],seed=2026091420,families={'test':128})
    fresh=dataset(fresh_cfg,'test');save(out/'fresh-config.json',fresh_cfg);save(out/'fresh-families.json',fresh[2])
    torch.save(dict(inputs=fresh[0].to(torch.uint8),labels=fresh[1]),out/'fresh-data.pt')
    fresh_hashes={m['family_sha256'] for m in fresh[2]}
    assert len(fresh_hashes)==128 and not fresh_hashes&set(h for x in allsteps[0] for h in x['fresh_family_hashes'])
    assert not fresh_hashes&allheldout
    for i,(name,cfg) in enumerate(zip(NAMES,configs)):
        p=ROOT/'results'/name;checkpoint=p/'checkpoint.pt';digest=sha(checkpoint)
        c=GPTNeoXConfig.from_dict(json.loads((p/'model-config.json').read_text()));c._attn_implementation='eager'
        model=GPTNeoXForCausalLM(c).float();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);install_binding_control(model,cfg)
        runs[i]['fresh_evaluation']=evaluate(model,fresh,cfg)
        runs[i]['fresh_ci']=helper.ci_family(runs[i]['fresh_evaluation']['predictions'],cfg['num_values'])
        assert digest==sha(checkpoint)
        existing=ROOT/'results/recall-supervision-pair-v0'/(name+'-query-probe.json')
        qp=json.loads(existing.read_text(encoding='utf-8')) if existing.exists() else helper.query_probe(p,cfg)
        assert qp['checkpoint_sha256']==digest
        save(out/(name+'-query-probe.json'),qp)
        runs[i]['query_probe']={k:v for k,v in qp.items() if k not in ['predictions','labels']}
    # Interval for paired family-level fresh-set accuracy difference, not independent examples.
    base=(np.array(runs[0]['fresh_evaluation']['predictions']).reshape(-1,8)==np.arange(8)).mean(1)
    rng=np.random.default_rng(2026091421);idx=rng.integers(0,128,(4000,128))
    for r in runs:
        scores=(np.array(r['fresh_evaluation']['predictions']).reshape(-1,8)==np.arange(8)).mean(1);means=(scores-base)[idx].mean(1)
        r['fresh_paired_delta_ci']={'mean':float((scores-base).mean()),'low':float(np.quantile(means,.025)),'high':float(np.quantile(means,.975))}
    result=dict(started_utc=started,finished_utc=now(),runs=runs,audit=audits,
        pairing=dict(initialization_equal=True,all_4000_training_batches_same=True,source_changes='Optional binding hook added after supervision pair; disabled for normal models; data and query helper hashes unchanged'),
        fresh_family_count=128,fresh_data_seed=2026091420,optimizer_updates=16000,input_tokens=sum(r['input_tokens'] for r in runs),
        supervised_answers=sum(r['supervised_answer_tokens'] for r in runs),gpu_jobs_started=0,
        claim_scope='Four adaptive single-seed 330k/128-token local diagnostic runs; no new algorithm, no training-seed uncertainty estimate, no cloud billing assertion')
    save(out/'result.json',result);save(ROOT/'logs/recall-followup-audit.json',dict(utc=now(),audits=audits,pairing=result['pairing'],fresh_families=128))
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for r,label in zip(runs,LABELS):axes[0].plot([d['step'] for d in r['development']],[d['accuracy']*100 for d in r['development']],marker='o',markersize=3,label=label)
    axes[0].axhline(80,color='grey',ls='--');axes[0].set(xlabel='Updates',ylabel='Development accuracy (%)',ylim=(0,105));axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    vals=[r['fresh_ci']['accuracy']*100 for r in runs];err=np.array([[100*(r['fresh_ci']['accuracy']-r['fresh_ci']['low']) for r in runs],[100*(r['fresh_ci']['high']-r['fresh_ci']['accuracy']) for r in runs]])
    axes[1].bar(range(4),vals,yerr=err,capsize=4,color=['#4878c0','#ee854a','#6acc64','#d65f5f']);axes[1].set_xticks(range(4),['Last 3e-4','Four 3e-4','Predecessor','Last 1e-3'],rotation=18);axes[1].set(ylim=(0,110),ylabel='Fresh-family accuracy (%)',title='128 new families; family bootstrap intervals')
    for i,v in enumerate(vals):axes[1].text(i,v+8,f'{v:.1f}%',ha='center')
    fig.tight_layout();fig.savefig(out/'results.png',dpi=170);plt.close(fig)
    snapshot=out/'source-snapshot';snapshot.mkdir()
    for f in [Path(__file__),ROOT/'scripts/report-recall-supervision.py',ROOT/'src/run_recall_learnability.py',ROOT/'src/recall_binding.py',ROOT/'src/recall_supervision.py']:shutil.copy2(f,snapshot/f.name)
    save(out/'manifest.json',[dict(path=str(p.relative_to(out)),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()])
    print(json.dumps(dict(results=[dict(name=r['name'],dev=r['development'][-1]['accuracy'],test=r['dense_test']['accuracy'],fresh=r['fresh_evaluation']['accuracy'],query_flip=r['query_probe']['prediction_changed_fraction'],gate=r['learnability_gate_passed']) for r in runs],updates=result['optimizer_updates'])))

if __name__=='__main__':main()
