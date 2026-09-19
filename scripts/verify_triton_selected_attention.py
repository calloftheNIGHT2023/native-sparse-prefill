"""Required CUDA gate before any fused selected-attention timing or training."""
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
sys.path.insert(0, str(ROOT / 'src'))


def main(a):
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat(); tick = time.perf_counter(); checks = []
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
               ['src/triton_selected_attention.py', 'src/chunked_topk_attention.py',
                'scripts/verify_triton_selected_attention.py']}
    for name in sources:
        dst = out/'source'/name; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT/name).read_bytes())
    try:
        import triton
        from triton_selected_attention import triton_selected_attention
        from chunked_topk_attention import select_causal_topk
        assert torch.cuda.is_available(), 'CUDA required; no fallback'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(2026091531)
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for length, dim, slots in [(1, 16, 1), (7, 19, 8), (37, 32, 8), (61, 128, 8), (41, 64, 32)]:
                for dropout in [0., .2]:
                    # Strided Q/K/V tests the actual interleaved QKV storage layout.
                    packed = torch.randn(2, length, 3, dim, device='cuda', dtype=dtype)
                    q, k, v = [t.detach().requires_grad_(True) for t in packed.unbind(2)]
                    ids = select_causal_topk(q, k, slots, min(2, slots), 11)
                    keep = torch.rand(ids.shape, device='cuda') >= dropout
                    actual = triton_selected_attention(q, k, v, ids, dropout, keep)
                    mask = torch.zeros(2, length, length, device='cuda', dtype=torch.bool)
                    drop = torch.zeros_like(mask, dtype=torch.float32)
                    for g in range(2):
                        for row in range(length):
                            ok = ids[g, row] >= 0
                            mask[g, row, ids[g, row, ok]] = True
                            drop[g, row, ids[g, row, ok]] = keep[g, row, ok].float() / (1 - dropout)
                    score = q.float() @ (k.float() / dim**.5).transpose(1, 2)
                    expected = ((score.masked_fill(~mask, -torch.inf).softmax(-1) * drop) @ v.float()).to(dtype)
                    go = torch.randn_like(actual)
                    da = torch.autograd.grad(actual, (q, k, v), go)
                    db = torch.autograd.grad(expected, (q, k, v), go)
                    atol = 3e-5 if dtype == torch.float32 else (4e-3 if dtype == torch.bfloat16 else 7e-4)
                    rtol = 3e-4 if dtype == torch.float32 else .02
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                    for u, w in zip(da, db):
                        torch.testing.assert_close(u, w, atol=atol, rtol=rtol)
                    check = dict(dtype=str(dtype), length=length, dim=dim, slots=slots, dropout=dropout,
                        output_max_abs=float((actual-expected).abs().max().detach()),
                        gradient_max_abs=max(float((u-w).abs().max()) for u, w in zip(da, db)), status='passed')
                    checks.append(check); print(json.dumps(check), flush=True)
        status = 'passed'; error = None
    except Exception as exc:
        status = 'failed'; error = dict(message=str(exc), traceback=traceback.format_exc())
    result = dict(status=status, started_utc=start, finished_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-tick, checks=checks, error=error, torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, sources=sources,
        scientific_optimizer_updates=0, limitation='Fixed-support gate; no model-quality or selector-precision claim')
    (out/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if status != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True, type=Path); main(p.parse_args())
