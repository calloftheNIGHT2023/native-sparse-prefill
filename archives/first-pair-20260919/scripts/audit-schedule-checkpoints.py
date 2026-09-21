"""Independent final-weight evaluation after wrapper/log anomaly; zero training."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from zoology_entry import LanguageModel
from zoology.config import ModelConfig
from zoology_sparse_schedule import install,set_epoch
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
torch.set_num_threads(4);run=ROOT/'results/schedule-screen-cloud-v0';out=ROOT/'results/schedule-checkpoint-audit-v0';out.mkdir(exist_ok=False)
data=torch.load(run/'shared-data-and-initialization.pt',map_location='cpu',weights_only=True)['splits']['fresh'];rows=[];epochs=[]
for summary in read(run/'result.json')['runs']:
    method=summary['method'];folder=run/method;p=folder/'checkpoint.pt';digest=sha(p);state=torch.load(p,map_location='cpu',weights_only=False)
    model=LanguageModel(ModelConfig(**state['config']['model']));model.load_state_dict(state['model']);install(model,method);set_epoch(model,40);model.eval()
    predictions=[];total_loss=0.
    with torch.no_grad():
        for start in range(0,len(data['inputs']),32):
            x=data['inputs'][start:start+32];y=data['labels'][start:start+32];logits=model(x);mask=y!=-100
            predictions.extend(logits.argmax(-1)[mask].tolist());total_loss+=float(torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='sum'))
    saved=read(folder/'result.json')['fresh'];labels=data['labels'][data['labels']!=-100]
    accuracy=float((torch.tensor(predictions)==labels).double().mean());agreement=float((torch.tensor(predictions)==torch.tensor(saved['predictions'])).double().mean())
    assert agreement==1.0 and accuracy==saved['accuracy'] and sha(p)==digest
    rows.append(dict(method=method,checkpoint_sha256=digest,answers=len(predictions),saved_gpu_accuracy=saved['accuracy'],
        recomputed_cpu_accuracy=accuracy,prediction_agreement=agreement,cpu_nll=total_loss/len(predictions),saved_gpu_nll=saved['nll']))
    for line in (folder/'events.jsonl').read_text().splitlines():
        event=json.loads(line)
        if event['event']=='validation':epochs.append(dict(method=method,**event,source_file=str(folder/'events.jsonl'),source_sha256=sha(folder/'events.jsonl')))
save(out/'full-fresh-recomputation.json',dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,optimizer_updates=0,total_answers=20000,runs=rows))
save(out/'epoch-summary-reconstructed-from-events.json',dict(label='Derived summary, NOT original stdout log',epochs=epochs))
save(out/'wrapper-anomaly.json',dict(observed_wrapper_exit_code=125,expected_clean_exit=False,
    console_lines=190,console_last_epoch='second_warm8 epoch28',source_epoch_events=200,source_optimizer_steps=62600,
    five_run_results_complete=True,result_manifest_files_verified=56,final_checkpoint_scores_independently_recomputed=True,
    cause='Unknown. A fresh timeout 2s true self-check returned0; not evidence that original wrapper exited cleanly.',
    action='Preserve original stdout and wrapper status; use complete hashed per-step events and independently verified checkpoints for scientific results; derived epoch summary is labeled reconstructed.'))
save(out/'manifest.json',[dict(path=p.name,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]);print(json.dumps(rows))
