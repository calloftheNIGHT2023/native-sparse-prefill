"""Fixed-metadata backward timing; run original/candidate in separate processes.

Does not include routing, a full model, optimizer steps, or Python process startup.
Run ABBA order after correctness gates; no other GPU workload should be active.
"""
import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import flash_moba_cuda
from flash_moba.flash_moba_interface import flash_moba_attn_varlen_func, decide_lg_block_m

ROOT = Path(__file__).resolve().parents[1]


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    torch.set_num_threads(4)
    bundle = ROOT / 'data/flashmoba-backward-fixed-v0'
    z = torch.load(bundle / 'qkv.pt', weights_only=True)
    q, k, v = [z[n].cuda().requires_grad_() for n in ['q', 'k', 'v']]
    go = torch.load(bundle / 'upstream-gradient.pt', weights_only=True).cuda()
    meta = torch.load(a.metadata, weights_only=True)
    offsets, counts, indices = [meta[n].cuda() for n in ['offsets', 'counts', 'indices']]
    n = q.shape[0]
    cu = torch.tensor([0, n], device='cuda', dtype=torch.int32)
    rows = []
    for deterministic in [False, True]:
        output = flash_moba_attn_varlen_func(
            q, k, v, cu, cu, n, n, offsets, counts, indices,
            decide_lg_block_m(4, 128, n, True), 128,
            dropout_p=0., causal=True, deterministic=deterministic)
        def backward():
            return torch.autograd.grad(output, (q, k, v), go, retain_graph=True)
        for _ in range(50):
            backward()
        torch.cuda.synchronize()
        samples = []
        for batch in range(9):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(50):
                backward()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / 50)
        rows.append(dict(deterministic=deterministic, milliseconds_per_backward=samples,
                         median_ms=statistics.median(samples)))
        del output
    result = dict(status='complete', label=a.label, started_utc=start,
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter()-tick, rows=rows,
                  extension_path=flash_moba_cuda.__file__,
                  extension_sha256=hashlib.sha256(Path(flash_moba_cuda.__file__).read_bytes()).hexdigest(),
                  metadata_sha256=hashlib.sha256(a.metadata.read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  scientific_optimizer_updates=0,
                  scope='CUDA-event repeated eager backward on one frozen 8K layer graph; includes launch gaps and allocations, excludes routing and full-model training.')
    (a.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    (a.output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--metadata', type=Path, default=ROOT/'results/flashmoba-long-backward-original-v0/actual-routing.pt')
    main(p.parse_args())
