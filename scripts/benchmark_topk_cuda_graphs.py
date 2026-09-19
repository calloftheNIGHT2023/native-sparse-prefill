"""Fair eager/graph diagnostic for exact top-k and dense CUDA attention.

Graphs remove Python dispatch from BOTH methods. Sparse totals include selection;
cached-index timings are labelled components. No optimizer or model-quality run.
Capture/compilation cost is reported separately, not hidden in steady-state time.
"""
import argparse
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chunked_topk_attention import ChunkedTopKAttention, select_causal_topk, selected_attention
from benchmark_chunked_topk import DenseSDPA


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def capture(fn, warmup=3):
    """Warm up on a side stream and retain every returned graph output."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    torch.cuda.synchronize()
    return graph, output


def invoke(model, x, go, backward):
    if backward:
        y = model(x)
        dx, = torch.autograd.grad(y, x, go)
        return y, dx
    with torch.no_grad():
        return (model(x),)


def correctness(include_triton=False, include_triton_rank=False, include_streaming=False):
    """Changed inputs must update the selector, output and gradient on replay."""
    rows = []
    for dtype in [torch.float32, torch.bfloat16]:
        for method in (['chunked_topk', 'dense_sdpa_auto'] + (['triton_topk'] if include_triton else [])
                       + (['triton_rank_topk'] if include_triton_rank else [])
                       + (['streaming_topk'] if include_streaming and dtype==torch.bfloat16 else [])):
            torch.manual_seed(2026091509)
            x = torch.randn(2, 37, 3, 2, 16, device='cuda', dtype=dtype, requires_grad=True)
            go = torch.randn(2, 37, 2, 16, device='cuda', dtype=dtype)
            if method == 'streaming_topk':
                from triton_streaming_selector import StreamingTopKAttention
                model = StreamingTopKAttention(dropout=0, query_chunk=11).train()
            elif method == 'triton_rank_topk':
                from triton_topk_selector import TritonRankTopKAttention
                model = TritonRankTopKAttention(dropout=0, query_chunk=11).train()
            elif method == 'triton_topk':
                from triton_selected_attention import TritonTopKAttention
                model = TritonTopKAttention(dropout=0, query_chunk=11).train()
            else:
                model = (ChunkedTopKAttention(dropout=0, query_chunk=11) if method == 'chunked_topk'
                         else DenseSDPA(0)).train()
            fn = lambda: invoke(model, x, go, True)
            graph, actual = capture(fn)
            errors = []
            for trial in range(2):
                with torch.no_grad():
                    x.copy_(torch.randn_like(x))
                    go.copy_(torch.randn_like(go))
                expected = fn()
                graph.replay()
                torch.cuda.synchronize()
                for observed, reference in zip(actual, expected):
                    torch.testing.assert_close(observed, reference, rtol=3e-3 if dtype == torch.bfloat16 else 5e-5,
                                               atol=3e-3 if dtype == torch.bfloat16 else 5e-5)
                errors.append(dict(output_max_abs=float((actual[0] - expected[0]).abs().max().detach()),
                                   gradient_max_abs=float((actual[1] - expected[1]).abs().max())))
            rows.append(dict(method=method, dtype=str(dtype), changed_input_checks=errors, status='passed'))
            del graph, actual, expected, model, x, go, fn

    q, k = [torch.randn(2, 61, 16, device='cuda') for _ in range(2)]
    graph, indices = capture(lambda: select_causal_topk(q, k, 8, 2, 13))
    old = indices.clone()
    q.copy_(torch.randn_like(q)); k.copy_(torch.randn_like(k))
    graph.replay(); torch.cuda.synchronize()
    expected = select_causal_topk(q, k, 8, 2, 13)
    assert torch.equal(indices, expected), 'Graph froze or changed the selector'
    changed = int((indices != old).sum())
    assert changed > 0
    rows.append(dict(method='selector', changed_index_slots=changed, status='passed'))
    del graph, q, k, indices, expected, old
    gc.collect(); torch.cuda.empty_cache()
    return rows


def main(a):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    if min(a.lengths + [a.batch, a.heads, a.dim, a.chunk, a.repeats, a.iterations]) < 1:
        raise ValueError('Positive shapes and repeats required')
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = [Path(__file__), ROOT / 'scripts/benchmark_chunked_topk.py',
               ROOT / 'src/chunked_topk_attention.py', ROOT / 'src/zoology_sparse_schedule.py']
    if a.include_triton:
        if a.triton_gate is None:
            raise ValueError('--include-triton requires --triton-gate')
        gate = json.loads(a.triton_gate.read_text(encoding='utf-8'))
        assert gate['status'] == 'passed' and len(gate['checks']) == 30, 'Incomplete Triton correctness gate'
        assert gate['sources']['src/triton_selected_attention.py'] == sha(ROOT/'src/triton_selected_attention.py')
        assert gate['gpu'] == torch.cuda.get_device_name(), 'Gate belongs to another GPU type'
        sources.append(ROOT/'src/triton_selected_attention.py')
    if a.include_triton_rank:
        assert a.include_triton and a.selector_gate is not None, 'Both Triton gates required'
        gate = json.loads(a.selector_gate.read_text(encoding='utf-8'))
        assert gate['status'] == 'passed' and len(gate['checks']) == 15
        assert gate['sources']['src/triton_topk_selector.py'] == sha(ROOT/'src/triton_topk_selector.py')
        assert gate['gpu'] == torch.cuda.get_device_name()
        sources.append(ROOT/'src/triton_topk_selector.py')
    if a.include_streaming:
        assert a.include_triton_rank and a.streaming_gate is not None and a.dtype=='bfloat16'
        gate=json.loads(a.streaming_gate.read_text(encoding='utf-8'))
        assert gate['status']=='passed' and len(gate['checks'])==8
        assert gate['sources']['src/triton_streaming_selector.py']==sha(ROOT/'src/triton_streaming_selector.py')
        sources.append(ROOT/'src/triton_streaming_selector.py')
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    for p in sources:
        dest = out / 'source' / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
    environment = dict(torch=torch.__version__, python=sys.version, platform=platform.platform(),
                       cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                       free_total_bytes=list(torch.cuda.mem_get_info()), exclusive_gpu_access_verified=False)
    try:
        environment['gpu_snapshot'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu',
            '--format=csv,noheader'], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception as exc:
        environment['gpu_snapshot_error'] = str(exc)
    save(out / 'manifest.json', dict(utc=utc(), environment=environment, sources=hashes,
         arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}))
    start = utc(); clock = time.perf_counter(); rows = []; checks = []

    def event(data):
        with (out / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(utc=utc(), **data)) + '\n')

    def deadline():
        if time.perf_counter() - clock > a.max_seconds:
            raise TimeoutError('Wall cap checked between operations')

    def measure(fn):
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tick = time.perf_counter(); begin.record()
        for _ in range(a.iterations):
            fn()
        end.record(); end.synchronize()
        return dict(gpu_ms=begin.elapsed_time(end) / a.iterations,
                    wall_ms=1000 * (time.perf_counter() - tick) / a.iterations)

    try:
        checks = correctness(a.include_triton, a.include_triton_rank, a.include_streaming); save(out / 'correctness.json', checks)
        for length in a.lengths:
            deadline()
            dtype = getattr(torch, a.dtype)
            x = torch.randn(a.batch, length, 3, a.heads, a.dim, device='cuda', dtype=dtype, requires_grad=True)
            go = torch.randn(a.batch, length, a.heads, a.dim, device='cuda', dtype=dtype)
            q, k, v = [z.detach().permute(0, 2, 1, 3).reshape(a.batch*a.heads, length, a.dim) for z in x.unbind(2)]
            ids = select_causal_topk(q, k, 8, 2, a.chunk)
            methods = dict(chunked_topk=ChunkedTopKAttention(dropout=a.dropout, query_chunk=a.chunk),
                           dense_sdpa_auto=DenseSDPA(a.dropout), dense_sdpa_flash=DenseSDPA(a.dropout, True))
            if a.include_triton:
                from triton_selected_attention import TritonTopKAttention, triton_selected_attention
                methods['triton_topk'] = TritonTopKAttention(dropout=a.dropout, query_chunk=a.chunk)
            if a.include_triton_rank:
                from triton_topk_selector import TritonRankTopKAttention, select_triton_topk
                methods['triton_rank_topk'] = TritonRankTopKAttention(dropout=a.dropout, query_chunk=a.chunk)
            if a.include_streaming:
                from triton_streaming_selector import StreamingTopKAttention, select_streaming_topk
                methods['streaming_topk']=StreamingTopKAttention(dropout=a.dropout,query_chunk=a.chunk)
            cases = [(method, operation) for method in methods for operation in
                     ['prefill_core', 'attention_forward_backward']]
            cases += [('selector_only', 'component'), ('selected_aggregation_cached_indices', 'component')]
            if a.include_triton:
                cases.append(('triton_aggregation_cached_indices', 'component'))
            if a.include_triton_rank:
                cases.append(('triton_selector_only', 'component'))
            if a.include_streaming:
                cases.append(('streaming_selector_only','component'))
            for method, operation in cases:
                deadline()
                row = dict(method=method, operation=operation, length=length, batch=a.batch, heads=a.heads,
                           dim=a.dim, dtype=a.dtype, chunk=a.chunk,
                           dropout=a.dropout if operation == 'attention_forward_backward' else 0,
                           selection_included=method in ['chunked_topk', 'triton_topk', 'triton_rank_topk',
                                                         'selector_only', 'triton_selector_only',
                                                         'streaming_topk', 'streaming_selector_only'],
                           diagnostic_only=operation == 'component')
                graph = None; outputs = None
                try:
                    if method == 'selector_only':
                        fn = lambda: (select_causal_topk(q, k, 8, 2, a.chunk),)
                    elif method == 'triton_selector_only':
                        fn = lambda: (select_triton_topk(q, k, 8, 2, a.chunk),)
                    elif method == 'streaming_selector_only':
                        fn = lambda: (select_streaming_topk(q,k),)
                    elif method == 'selected_aggregation_cached_indices':
                        fn = lambda: (selected_attention(q, k, v, ids, query_chunk=a.chunk),)
                    elif method == 'triton_aggregation_cached_indices':
                        fn = lambda: (triton_selected_attention(q, k, v, ids),)
                    else:
                        model = methods[method]; model.train(operation == 'attention_forward_backward')
                        fn = lambda: invoke(model, x, go, operation == 'attention_forward_backward')
                    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    tick = time.perf_counter()
                    graph, outputs = capture(fn)
                    row['capture_and_warmup_seconds'] = time.perf_counter() - tick
                    row['capture_baseline_allocated_bytes'] = baseline
                    row['capture_peak_increment_bytes'] = torch.cuda.max_memory_allocated() - baseline
                    row['post_capture_allocated_increment_bytes'] = torch.cuda.memory_allocated() - baseline
                    if operation == 'attention_forward_backward' and a.dropout:
                        graph.replay(); torch.cuda.synchronize(); first = outputs[0].clone()
                        graph.replay(); torch.cuda.synchronize()
                        row['dropout_changes_between_replays'] = bool((first != outputs[0]).any())
                        assert row['dropout_changes_between_replays'], 'Dropout RNG frozen by graph'
                        assert all(bool(torch.isfinite(t).all()) for t in outputs)
                        del first
                    for _ in range(3):
                        fn(); graph.replay()
                    torch.cuda.synchronize()
                    samples = dict(eager=[], graph=[])
                    for repeat in range(a.repeats):
                        deadline()
                        for mode in (['eager', 'graph'] if repeat % 2 == 0 else ['graph', 'eager']):
                            observed = measure(fn if mode == 'eager' else graph.replay)
                            samples[mode].append(observed)
                            event(dict(event='sample', case=row, repeat=repeat, mode=mode, **observed))
                    row['samples'] = samples
                    for mode in samples:
                        row[mode] = {metric: statistics.median([s[metric] for s in samples[mode]])
                                     for metric in ['gpu_ms', 'wall_ms']}
                    row['graph_over_eager_gpu_ratio'] = row['graph']['gpu_ms'] / row['eager']['gpu_ms']
                    if method.startswith('dense_sdpa'):
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                            fn(); torch.cuda.synchronize()
                        row['observed_attention_operators'] = sorted({e.key for e in prof.key_averages()
                                                                     if 'attention' in e.key.lower()})
                    row['status'] = 'passed'
                except TimeoutError:
                    raise
                except Exception as exc:
                    row.update(status='unsupported_or_failed', error=str(exc), traceback=traceback.format_exc())
                rows.append(row); event(dict(event='case_complete', result=row))
                print(json.dumps({k: value for k, value in row.items() if k not in ['samples', 'traceback']}), flush=True)
                del graph, outputs, fn
                gc.collect(); torch.cuda.empty_cache()
            del x, go, q, k, v, ids, methods
        status = 'complete'
    except Exception as exc:
        status = 'incomplete'; event(dict(event='failure', error=str(exc), traceback=traceback.format_exc()))
    result = dict(status=status, started_utc=start, finished_utc=utc(), wall_seconds=time.perf_counter()-clock,
                  environment=environment, correctness=checks, rows=rows, scientific_optimizer_updates=0,
                  sources_unchanged=all(sha(ROOT / p) == h for p, h in hashes.items()),
                  limitations=['Random attention-core tensors, not whole-model quality or training',
                    'Exact selector remains quadratic; cached-index components excluded from total claims',
                    'Graph construction and input transfer excluded from steady-state timing',
                    'Both dense and sparse use graphs; no graph-vs-eager cross-method comparison',
                    'GPU shared with desktop applications; timing is diagnostic',
                    'Graph memory includes persistent graph pools; not comparable to prior eager-only peak tables'])
    save(out / 'benchmark.json', result)
    if status != 'complete' or not result['sources_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    p.add_argument('--lengths', nargs='+', type=int, default=[1024, 4096])
    p.add_argument('--batch', type=int, default=1); p.add_argument('--heads', type=int, default=1)
    p.add_argument('--dim', type=int, default=128); p.add_argument('--chunk', type=int, default=256)
    p.add_argument('--dropout', type=float, default=.1); p.add_argument('--repeats', type=int, default=15)
    p.add_argument('--iterations', type=int, default=3); p.add_argument('--max-seconds', type=float, default=180)
    p.add_argument('--seed', type=int, default=2026091521)
    p.add_argument('--include-triton', action='store_true')
    p.add_argument('--triton-gate', type=Path)
    p.add_argument('--include-triton-rank', action='store_true')
    p.add_argument('--selector-gate', type=Path)
    p.add_argument('--include-streaming',action='store_true')
    p.add_argument('--streaming-gate',type=Path)
    main(p.parse_args())
