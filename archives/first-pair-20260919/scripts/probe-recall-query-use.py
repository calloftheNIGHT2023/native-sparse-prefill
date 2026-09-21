"""Read-only development-set diagnosis; no checkpoint selection or updates."""
import hashlib,json,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from far_recall import symbolic_lookup
from run_recall_learnability import dataset,prediction

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')

def main():
    started=now(); timer=time.perf_counter(); torch.set_num_threads(4)
    folder=ROOT/'results/far-recall-cpu-stream-v2'
    out=ROOT/'results/far-recall-query-use-v0'; out.mkdir(exist_ok=False)
    cfg=json.loads((folder/'frozen-config.json').read_text())
    checkpoint=folder/'checkpoint.pt'; before=sha(checkpoint)
    c=GPTNeoXConfig.from_dict(json.loads((folder/'model-config.json').read_text()))
    c._attn_implementation='eager'
    model=GPTNeoXForCausalLM(c).float()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']); model.eval()
    x,y,metas=dataset(cfg,'validation'); inputs=[]; labels=[]; groups=[]; mode=[]; first=[]; last=[]
    for fi,meta in enumerate(metas):
        for variant in range(cfg['num_values']):
            row=x[fi*cfg['num_values']+variant]
            ps=sorted(meta['record_positions']); vals=[int(row[p+2])-cfg['value_base'] for p in ps]
            mode.append(max(range(cfg['num_values']),key=lambda v: (vals.count(v),-v)))
            first.append(vals[0]); last.append(vals[-1])
            for p in ps:
                changed=row.clone(); changed[-1]=row[p+1]
                inputs.append(changed); labels.append(int(row[p+2])-cfg['value_base'])
                groups.append(dict(family=fi,variant=variant,query_record_position=p))
    xx=torch.stack(inputs); yy=torch.tensor(labels)
    assert torch.equal(symbolic_lookup(xx,cfg),yy)
    preds=[]; logits=[]
    with torch.no_grad():
        for start in range(0,len(xx),cfg['batch_size']):
            out_logits=prediction(model,xx[start:start+cfg['batch_size']])[:,cfg['value_base']:cfg['value_base']+cfg['num_values']]
            logits.append(out_logits); preds.extend(out_logits.argmax(-1).tolist())
    pp=torch.tensor(preds).reshape(-1,cfg['records']); truth=yy.reshape_as(pp)
    flips=[]; joint=[]
    for p,t in zip(pp,truth):
        for a in range(len(p)):
            for b in range(a+1,len(p)):
                if t[a]!=t[b]:
                    flips.append(bool(p[a]!=p[b])); joint.append(bool(p[a]==t[a] and p[b]==t[b]))
    def acc(p): return float((torch.tensor(p)==y).float().mean())
    result=dict(status='complete',started_utc=started,finished_utc=now(),seconds=time.perf_counter()-timer,
        checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=before,
        original_dev_accuracy=json.loads((folder/'result.json').read_text())['development'][-1]['accuracy'],
        original_dev_query_blind_baselines=dict(most_frequent_value_ties_lowest=acc(mode),first_record_value=acc(first),last_record_value=acc(last)),
        query_sweep_accuracy=float((pp==truth).float().mean()),
        different_answer_query_pairs=len(flips),prediction_changed_fraction=sum(flips)/len(flips),both_answers_correct_fraction=sum(joint)/len(joint),
        input_families=len(metas),base_examples=len(x),query_sweep_examples=len(xx),
        interpretation='Exploratory development intervention on one trained 330k model. Change only query key, keep records fixed. Correlated variants/pairs, not independent trials; no significance claim. Query-blind baselines read evidence values but ignore keys.',
        optimizer_updates=0,gpu_jobs_started=0)
    assert before==sha(checkpoint)
    save(out/'result.json',result); save(out/'predictions.json',dict(groups=groups,predictions=preds,labels=labels))
    snapshot=out/'source-snapshot'; snapshot.mkdir()
    for p in [Path(__file__),ROOT/'src/far_recall.py',ROOT/'src/run_recall_learnability.py']:
        shutil.copy2(p,snapshot/p.name)
    save(out/'manifest.json',[dict(path=str(p.relative_to(out)),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()])
    print(json.dumps(result))

if __name__=='__main__': main()
