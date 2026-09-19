"""Read-only value-swap intervention, fixed while the length128 run was training."""
import hashlib,json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from zoology_entry import LanguageModel
from zoology.config import ModelConfig
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    now=lambda:datetime.now(timezone.utc).isoformat();started=now();torch.set_num_threads(4)
    run=ROOT/'results/zoology-length128-cpu-v0';out=ROOT/'results/zoology-length128-source-values-v0';out.mkdir(exist_ok=False)
    checkpoint=run/'checkpoint.pt';digest=sha(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=LanguageModel(ModelConfig(**state['config']['model']));model.load_state_dict(state['model']);model.eval()
    data=torch.load(run/'fresh-evaluation-data.pt',weights_only=True);original=data['inputs'];changed=original.clone()
    positions=[];swap_positions=[];new_labels=[];old_labels=[];gaps=[]
    for i,(x,y) in enumerate(zip(original,data['labels'])):
        q=int(torch.where(y!=-100)[0][-1]);key=int(x[q]);positions.append(q);old_labels.append(int(y[q]))
        first=next(p+1 for p in range(0,8,2) if int(x[p])==key);second=((first-1+2)%8)+1
        changed[i,first]=x[second];changed[i,second]=x[first]
        new_labels.append(int(x[second]));swap_positions.append([first,second]);gaps.append(q-first)
        assert old_labels[-1]!=new_labels[-1]
        assert int((changed[i]!=x).sum())==2 and torch.equal(changed[i,8:],x[8:])
    predictions=[]
    with torch.no_grad():
        for start in range(0,len(original),32):
            logits=model(changed[start:start+32]);qs=torch.tensor(positions[start:start+32])
            predictions.extend(logits[torch.arange(len(qs)),qs].argmax(-1).tolist())
    earlier=json.loads((ROOT/'results/zoology-length128-query-v0/predictions.json').read_text())
    earlier_result=json.loads((ROOT/'results/zoology-length128-query-v0/result.json').read_text())
    assert earlier_result['checkpoint_sha256']==digest and earlier['query_positions']==positions and earlier['old_labels']==old_labels
    before=torch.tensor(earlier['original_predictions']);after=torch.tensor(predictions)
    old=torch.tensor(old_labels);new=torch.tensor(new_labels);far=torch.tensor(gaps)>64
    result=dict(started_utc=started,finished_utc=now(),examples=len(old),checkpoint_sha256=digest,
        original_accuracy=float((before==old).double().mean()),swapped_values_accuracy=float((after==new).double().mean()),
        prediction_changed_fraction=float((before!=after).double().mean()),both_correct_fraction=float(((before==old)&(after==new)).double().mean()),
        far_examples=int(far.sum()),far_original_accuracy=float((before[far]==old[far]).double().mean()) if far.any() else None,
        far_swapped_values_accuracy=float((after[far]==new[far]).double().mean()) if far.any() else None,
        optimizer_updates=0,scope='Supplementary diagnostic fixed during training before final fresh scores: swap two prefix values, keep every query/filler unchanged; reuse original predictions from same checkpoint. Same test examples and training seed, not independent confirmation.')
    assert sha(checkpoint)==digest
    save(out/'result.json',result)
    save(out/'predictions.json',dict(query_positions=positions,swapped_value_positions=swap_positions,gaps=gaps,
        original_predictions=before.tolist(),changed_predictions=after.tolist(),old_labels=old_labels,new_labels=new_labels))
    shutil.copy2(__file__,out/Path(__file__).name)
    save(out/'manifest.json',[dict(path=p.name,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()])
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
