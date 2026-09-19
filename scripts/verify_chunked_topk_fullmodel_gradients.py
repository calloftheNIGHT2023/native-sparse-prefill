"""Full-model first-order gradient equivalence, using artificial audit targets.

No optimizer step and no scientific training. Evaluation mode disables dropout;
FP64 isolates the operator chain from the documented FP32 routing discontinuity.
"""
import hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from router_author_control import make_model,selected_logits
from zoology_sparse_schedule import install
from chunked_topk_attention import install_chunked

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()

def main():
    out=ROOT/'results/chunked-topk-fullmodel-gradients-v0';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);tick=time.perf_counter();started=utc()
    paths=[ROOT/'results/router-author-falsification-lowlr-v0/checkpoint.pt',
           ROOT/'results/router-author-falsification-confirm-v0/checkpoint.pt']
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    data=torch.load(ROOT/'results/router-author-falsification-evaluation-v0/evaluation-data.pt',map_location='cpu',weights_only=True)['test']
    checks=[]
    for path in paths:
        c=torch.load(path,map_location='cpu',weights_only=False)
        a=make_model(c['config'],c['model']).double();b=make_model(c['config'],c['model']).double()
        install(a,'native');install_chunked(b);a.eval();b.eval()
        losses=[]
        for m in [a,b]:
            logits,targets=selected_logits(m,data['inputs'][:2],data['labels'][:2])
            loss=torch.nn.functional.cross_entropy(logits,(targets+1)%8192)
            losses.append(float(loss.detach()));loss.backward()
        torch.testing.assert_close(torch.tensor(losses[0]),torch.tensor(losses[1]),rtol=1e-7,atol=1e-7)
        rows=[]
        for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
            assert name==other and torch.equal(p,q)
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-8)
            rows.append(dict(parameter=name,maximum_absolute_gradient_error=float((p.grad-q.grad).abs().max())))
        check=dict(checkpoint=str(path.relative_to(ROOT)),losses=losses,parameters=rows,
                   maximum_absolute_gradient_error=max(r['maximum_absolute_gradient_error'] for r in rows))
        checks.append(check);print(json.dumps({k:v for k,v in check.items() if k!='parameters'}),flush=True)
    assert all(sha(ROOT/p)==h for p,h in before.items())
    result=dict(status='passed',started_utc=started,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,
                checks=checks,checkpoint_hashes=before,dtype='float64',device='cpu',optimizer_updates=0,
                artificial_targets=True,dropout=0,scope='Technical gradient-chain audit, not training or quality evidence',
                sources={p:sha(ROOT/p) for p in ['scripts/verify_chunked_topk_fullmodel_gradients.py','src/chunked_topk_attention.py']})
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
