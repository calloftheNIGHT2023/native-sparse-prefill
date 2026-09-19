"""GPU structural and independent numerical checks before scientific evaluation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
from pathlib import Path
import torch
from task_route_intervention import routed_attention as previous
from task_route_selective import routed_attention, select_heads

torch.manual_seed(2026091620)
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
n, start = 4096, 4000
q = torch.randn(n, 14, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(n, 2, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn_like(k)
native, mass, original_ids = routed_attention(q, k, v, .125, 'baseline', start, None, [3, 4], return_ids=True)
old = previous(q, k, v, .125, 'rebuild')
assert torch.equal(native, old)
heads = select_heads(mass)
checks = []
for active, forced in [(heads, [3, 4]), (heads, [5, 6]), ([0, 3, 7, 13], [3, 4]), (list(range(14)), [3, 4])]:
    y, _, ids = routed_attention(q, k, v, .125, 'forced', start, forced, [3, 4], active, True)
    other = [h for h in range(14) if h not in active]
    assert torch.equal(ids[:, other], original_ids[:, other])
    assert torch.equal(y[:start], native[:start])
    assert torch.equal(y[:, other], native[:, other])
    refs = []
    for head in range(14):
        ts = (ids[-1, head, :, None].long() * 128 + torch.arange(128, device='cuda')[None, :]).flatten()
        ref = torch.softmax(q[-1, head].float() @ k[ts, head // 7].float().T * .125, dim=-1) @ v[ts, head // 7].float()
        refs.append(ref)
    ref = torch.stack(refs)
    error = float((ref - y[-1].float()).abs().max())
    assert torch.allclose(ref, y[-1].float(), atol=.03, rtol=.03)
    if len(active) == 14:
        old_forced = previous(q, k, v, .125, 'forced', start=start, forced=forced)
        assert torch.equal(y, old_forced)
    checks.append(dict(heads=active, forced=forced, fp32_max_abs_error=error,
                       untouched_heads_exact=True, prefix_exact=True))
# Independent loop checks the GQA mapping and normalization used for head scoring.
mass_ref = []
for h in range(14):
    p = torch.softmax(q[-1, h].float() @ k[:, h // 7].float().T * .125, -1)
    mass_ref.append(float(p[384:640].sum()))
assert max(abs(a-b) for a,b in zip(mass,mass_ref)) < 1e-6
assert select_heads([.5] * 14) == [0,1,2,3]
out = Path('results/task-route-selective-preflight-v0')
out.mkdir(parents=True, exist_ok=False)
result = dict(status='passed', checks=checks, mass_ref_error=max(abs(a-b) for a,b in zip(mass,mass_ref)),
              optimizer_updates=0, scored_task_predictions=0)
(out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
