"""Read-only query-key intervention using the successful final upstream model."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from zoology_entry import configuration,LanguageModel
from zoology.config import ModelConfig
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(args):
    now=lambda:datetime.now(timezone.utc).isoformat();started=now();torch.set_num_threads(4)
    run=args.run;out=args.output;out.mkdir(exist_ok=False)
    checkpoint=run/'checkpoint.pt';digest=sha(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=LanguageModel(ModelConfig(**state['config']['model']));model.load_state_dict(state['model']);model.eval()
    data=torch.load(run/'fresh-evaluation-data.pt',weights_only=True);original=data['inputs'];changed=original.clone();positions=[];new_labels=[];old_labels=[]
    for i,(x,y) in enumerate(zip(original,data['labels'])):
        q=int(torch.where(y!=-100)[0][-1]);positions.append(q);key=int(x[q]);old_labels.append(int(y[q]))
        old_index=next(p for p in range(0,8,2) if int(x[p])==key);new_index=(old_index+2)%8
        changed[i,q]=x[new_index];new_labels.append(int(x[new_index+1]))
        assert old_labels[-1]!=new_labels[-1]
    def predict(x):
        predictions=[]
        with torch.no_grad():
            for start in range(0,len(x),32):
                logits=model(x[start:start+32]);q=torch.tensor(positions[start:start+32])
                predictions.extend(logits[torch.arange(len(q)),q].argmax(-1).tolist())
        return torch.tensor(predictions)
    before=predict(original);after=predict(changed);old=torch.tensor(old_labels);new=torch.tensor(new_labels)
    result=dict(started_utc=started,finished_utc=now(),examples=len(old),checkpoint_sha256=digest,source_run=str(run),sequence_length=original.shape[1],
        original_last_query_accuracy=float((before==old).double().mean()),changed_query_accuracy=float((after==new).double().mean()),
        prediction_changed_fraction=float((before!=after).double().mean()),both_correct_fraction=float(((before==old)&(after==new)).double().mean()),
        optimizer_updates=0,scope='Post-training intervention on fresh set: change only last supervised query key to another prefix key; duplicated query keys can be OOD. No selection or retraining.')
    assert sha(checkpoint)==digest
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(dict(query_positions=positions,original_predictions=before.tolist(),changed_predictions=after.tolist(),old_labels=old_labels,new_labels=new_labels)),encoding='utf-8')
    shutil.copy2(__file__,out/Path(__file__).name)
    (out/'manifest.json').write_text(json.dumps([dict(path=p.name,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()],indent=2),encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'results/zoology-basic-cpu-v1');p.add_argument('--output',type=Path,default=ROOT/'results/zoology-query-intervention-v0');main(p.parse_args())
