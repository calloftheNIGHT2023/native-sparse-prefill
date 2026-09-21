"""Independent CPU FP64 reference and replay of the saved full-model gradient comparisons."""
import json
from pathlib import Path
from datetime import datetime,timezone
import torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
folder=ROOT/'results/flashmoba-mixed-cost-real-qkv-v0'
x=torch.load(folder/'replay-tensors.pt',map_location='cpu',weights_only=True)
q,k,v=[x[n].double().requires_grad_() for n in ['q','k','v']]
n=q.shape[-2];mask=torch.ones(n,n,dtype=torch.bool).tril()
y=torch.softmax(((q@k.repeat_interleave(7,dim=1).transpose(-1,-2))/8).masked_fill(~mask,float('-inf')),dim=-1)@v.repeat_interleave(7,dim=1)
grad=torch.autograd.grad(y,(q,k,v),x['output_gradient'].double())
rows=[]
for mode in ['flash','sparse_allblocks']:
    actual=x[mode];torch.testing.assert_close(actual['output'].double(),y,atol=.03,rtol=.03)
    for a,b in zip(actual['gradients'],grad):torch.testing.assert_close(a.double(),b,atol=.03,rtol=.03)
    rows.append(dict(mode=mode,fp64_output_passed=True,fp64_qkv_gradients_passed=True,
                     output_relative_l2=float((actual['output'].double()-y).norm()/y.norm()),
                     gradient_relative_l2=[float((a.double()-b).norm()/b.norm()) for a,b in zip(actual['gradients'],grad)]))
d=ROOT/'results/flashmoba-mixed-cost-rounding-diagnostic-v0'
saved=torch.load(d/'comparison-gradients.pt',map_location='cpu',weights_only=True)
report=json.loads((d/'result.json').read_text(encoding='utf-8'));errors=[]
for p in report['pairs']:
    a=saved[p['reference']]['gradient'];b=saved[p['compared']]['gradient']
    errors.append(abs(float((b-a).norm()/a.norm())-p['gradient_relative_l2']))
assert max(errors)<2e-6
out=ROOT/'provenance/flashmoba-mixed-cost-cpu-replay-v0.json';assert not out.exists()
result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),torch=str(torch.__version__),
            cpu_reference='FP64 same-QKV causal GQA, same BF16 upstream gradient, atol=rtol=0.03',
            same_qkv_checks=rows,full_model_gradient_pairs_replayed=len(errors),max_pair_metric_abs_error=max(errors),
            scientific_optimizer_updates=0,scope='256-token first layer only; no 16K gradient or quality equivalence claim.')
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
