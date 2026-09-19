"""Check exact chain rule and uneven chunk weighting on an independent FP64 toy graph."""
import json
from pathlib import Path
from datetime import datetime,timezone
import torch
import torch.nn.functional as F
from chunked_lm_loss import chunked_head_backward
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/flashmoba-long-cost-cpu-gate-v0';OUT.mkdir(parents=True,exist_ok=False)
torch.manual_seed(2026091556);torch.set_num_threads(4)
x=torch.randn(1,19,9,dtype=torch.float64);w=torch.randn(9,11,dtype=torch.float64,requires_grad=True)
head=torch.nn.Linear(11,23,dtype=torch.float64);head.requires_grad_(False);target=torch.randint(23,(19,))
hidden=torch.tanh(x@w);loss=F.cross_entropy(head(hidden)[0],target);loss.backward();expected=w.grad.clone();rows=[]
for chunk in [1,7,19,32]:
    w.grad=None;h=torch.tanh(x@w);actual=chunked_head_backward(h,head,target,chunk)
    torch.testing.assert_close(actual,loss.detach(),atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(w.grad,expected,atol=1e-12,rtol=1e-10)
    rows.append(dict(chunk=chunk,loss_abs_error=float((actual-loss.detach()).abs()),gradient_max_abs_error=float((w.grad-expected).abs().max())))
r=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),checks=rows,scientific_optimizer_updates=0)
(OUT/'source.py').write_bytes(Path(__file__).read_bytes());(OUT/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
