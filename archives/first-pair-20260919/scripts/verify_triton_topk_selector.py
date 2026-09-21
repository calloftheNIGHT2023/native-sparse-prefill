"""Audit ranking against the same GEMM scores, including deterministic ties."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main(a):
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat(); tick = time.perf_counter(); checks = []
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
               ['src/triton_topk_selector.py', 'src/triton_selected_attention.py',
                'scripts/verify_triton_topk_selector.py']}
    for path in sources:
        dst=out/'source'/path; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes((ROOT/path).read_bytes())
    try:
        from triton_topk_selector import select_triton_topk
        from chunked_topk_attention import select_causal_topk
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(2026091541)
        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            for length, budget, chunk in [(1, 1, 1), (7, 8, 3), (37, 8, 11), (127, 16, 64), (1024, 8, 256)]:
                q, k = [torch.randn(2, length, 32, device='cuda', dtype=dtype) for _ in range(2)]
                if length == 37:
                    k[:, 3:15] = k[:, :1]  # Repeated keys deliberately create score ties.
                local = min(budget, 2); slots = min(length, budget); ls = min(slots, local)
                ids = select_triton_topk(q, k, budget, local, chunk)
                old = select_causal_topk(q, k, budget, local, chunk)
                differs = 0
                scaled = k / 32**.5
                for first in range(0, length, chunk):
                    end = min(first+chunk, length)
                    score = torch.bmm(q[:, first:end], scaled[:, :end].transpose(1, 2))
                    position = torch.arange(first, end, device='cuda')
                    allow = torch.arange(end, device='cuda')[None, :] <= position[:, None] - local
                    masked = score.masked_fill(~allow, -torch.inf)
                    # Stable sort is an independent reference for the documented tie rule.
                    order = torch.argsort(masked, dim=-1, descending=True, stable=True)
                    for group in range(2):
                        for offset, pos in enumerate(range(first, end)):
                            actual = ids[group, pos]; valid = actual[actual >= 0]
                            assert len(valid) == min(slots, pos+1) and len(valid.unique()) == len(valid)
                            assert bool((valid <= pos).all())
                            local_expected = torch.arange(pos, max(pos-ls, -1), -1, device='cuda')
                            assert torch.equal(actual[:len(local_expected)], local_expected)
                            rs = min(slots-ls, max(0, pos-local+1))
                            expected = order[group, offset, :rs]
                            assert torch.equal(actual[ls:ls+rs], expected), 'Incorrect selected ranking/tie rule'
                            assert bool((actual[ls+rs:] == -1).all())
                            a_set=valid.sort().values; b_set=old[group, pos][old[group, pos]>=0].sort().values
                            if not torch.equal(a_set, b_set):
                                differs += 1
                                # Top-k can differ only where the boundary score is tied.
                                remaining = max(0, pos-local+1)
                                assert rs > 0 and remaining > rs
                                ordered_values=masked[group, offset, order[group, offset]]
                                assert ordered_values[rs-1] == ordered_values[rs]
                checks.append(dict(dtype=str(dtype), length=length, budget=budget, chunk=chunk,
                    support_rows_different_from_torch_topk=differs, all_support_differences_at_ties=True, status='passed'))
                print(json.dumps(checks[-1]), flush=True)
        status='passed'; error=None
    except Exception as exc:
        status='failed'; error=dict(message=str(exc), traceback=traceback.format_exc())
    result=dict(status=status, started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-tick, checks=checks, error=error, sources=sources,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, scientific_optimizer_updates=0,
        limitation='Operator selection check, not model-quality evidence; BF16 ties may alter model predictions')
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if status != 'passed': raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); main(p.parse_args())
