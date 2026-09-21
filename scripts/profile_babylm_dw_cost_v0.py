"""Bounded D/W fixed-weight cost diagnosis; no optimizer and no scientific updates.

Default mode validates metadata only. CUDA execution requires --execute and an
external hard timeout. Stream-event timings include scheduling gaps and are not
pure kernel utilization. Component attribution is FORWARD ONLY.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SELF = 'scripts/profile_babylm_dw_cost_v0.py'
MAX_F, MAX_B, MAX_SECONDS = 64, 32, 900
TOLERANCE = {'atol': 1e-6, 'rtol': 1e-5}
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def validate(protocol):
    require(protocol.get('schema_version') == 1 and protocol.get('scope') == 'engineering_cost_diagnostic', 'Explicit engineering protocol required')
    require(protocol.get('device') == 'cuda' and protocol.get('dtype') == 'float32' and protocol.get('torch_num_threads') == 1, 'CUDA FP32 threads1 required')
    require(protocol.get('tolerance') == TOLERANCE, 'Original GPU atol/rtol must remain unchanged')
    require(0 < protocol['max_wall_seconds'] <= MAX_SECONDS, 'Wall cap exceeds 900 seconds')
    windows = protocol['window_indices']
    require(isinstance(windows, list) and 1 <= len(windows) <= 8 and len(set(windows)) == len(windows) and all(type(x) is int and x >= 0 for x in windows), 'One to eight unique fixed windows required')
    warmups = protocol.get('warmup_window_indices', [])
    require(isinstance(warmups, list) and len(warmups) <= 16 and all(x in windows for x in warmups), 'Warmups must use fixed selected windows')
    expected = {'forward_attempts': 2 * (len(warmups) + 2 * len(windows)), 'backward_attempts': 4 * len(windows), 'optimizer_updates': 0}
    require(expected['forward_attempts'] <= MAX_F and expected['backward_attempts'] <= MAX_B, 'Call budget exceeded')
    require(protocol.get('expected_calls') == expected, 'Expected fixed calls mismatch')
    require(set(protocol['arms']) == {'D', 'W'}, 'Only the fixed D/W pair is allowed')
    for arm in ['D', 'W']:
        value = protocol['arms'][arm]
        require(value.get('checkpoint_format', 'model_only') in {'model_only', 'full'}, 'Unknown checkpoint format')
        require(all(isinstance(value.get(k), str) and value[k] for k in ['checkpoint_path', 'checkpoint_sha256', 'checkpoint_protocol_sha256']), 'Checkpoint provenance fields required')
    sources = protocol['expected_source_hashes']
    required = [SELF, 'scripts/run_babylm_checkpoint_eval_v0.py', 'scripts/prepare_babylm_windows_v0.py', 'src/babylm_hybrid/config.py', 'src/babylm_hybrid/model.py', 'src/babylm_hybrid/attention.py', 'src/babylm_hybrid/evaluation.py', 'src/babylm_hybrid/local_attention_v0.py', 'src/babylm_hybrid/training.py']
    require(all(x in sources for x in required), 'Required source pin missing')
    for name, digest in sources.items():
        require(sha(resolve(name)) == digest, 'Source SHA differs: ' + name)
    require(isinstance(protocol.get('project_lock'), str) and protocol['project_lock'].startswith('/tmp/'), 'Linux project lock required')
    require(protocol.get('automatic_retry') is False, 'Automatic retry forbidden')
    require(protocol.get('shared_gpu_diagnostic_only') is True and protocol.get('min_free_gpu_bytes') == 16 * 1024 ** 3, 'Fixed shared-device diagnostic/free-memory contract required')
    require(set(protocol.get('execution_hardware', {})) == {'physical_uuid', 'driver_version'}, 'Exact physical GPU and driver contract required')
    require(isinstance(protocol.get('expected_runtime'), dict) and protocol['expected_runtime'].get('device') == 'cuda', 'Frozen runtime required')
    require(sha(resolve(protocol['train_manifest'])) == protocol['train_manifest_sha256'], 'Train manifest SHA differs')
    return expected


def gpu_snapshot(protocol):
    """Only device identity/utilization and numeric PIDs/memory, never argv/env."""
    identity = protocol['execution_hardware']
    base = ['nvidia-smi', '-i', identity['physical_uuid']]
    device = subprocess.run(base + ['--query-gpu=uuid,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
    fields = [x.strip() for x in device.stdout.strip().split(',')]
    require(len(fields) == 5 and fields[:2] == [identity['physical_uuid'], identity['driver_version']], 'nvidia-smi hardware identity mismatch')
    processes = subprocess.run(base + ['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=False)
    rows = []
    for line in processes.stdout.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) == 2 and parts[0].isdigit():
            rows.append({'pid': int(parts[0]), 'used_memory_mib': int(parts[1]) if parts[1].isdigit() else None})
    return {'utc': utc(), 'physical_uuid': fields[0], 'driver_version': fields[1],
            'memory_total_mib': int(fields[2]) if fields[2].isdigit() else None,
            'memory_used_mib': int(fields[3]) if fields[3].isdigit() else None,
            'utilization_gpu_percent': int(fields[4]) if fields[4].isdigit() else None,
            'compute_process_query_returncode': processes.returncode, 'observed_compute_processes': rows,
            'query_scope': 'Selected shared physical GPU; numeric PID/memory only, no command lines; not an isolation or whole-GPU idle claim.'}


def state_hash(torch, model):
    result = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().to('cpu').contiguous()
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        result[name] = {'dtype': str(tensor.dtype), 'shape': list(tensor.shape), 'sha256': hashlib.sha256(payload).hexdigest()}
    return result


def compare_tensor(torch, reference, actual):
    require(reference.shape == actual.shape and reference.dtype == actual.dtype, 'Tensor shape/dtype changed')
    r, a = reference.detach().to('cpu'), actual.detach().to('cpu')
    diff = (r - a).abs()
    tolerance = TOLERANCE['atol'] + TOLERANCE['rtol'] * r.abs()
    finite = bool(torch.isfinite(r).all() and torch.isfinite(a).all())
    return {'passed': finite and bool((diff <= tolerance).all()), 'max_abs': (float(diff.max()) if diff.numel() else 0.0) if finite else None,
            'max_tolerance_ratio': float((diff / tolerance).max()) if diff.numel() and finite else (0.0 if finite else None)}


class ForwardTimers:
    """Nonoverlapping modules only; hooks return None and never alter tensors."""
    def __init__(self, torch, model):
        self.torch, self.model = torch, model
        self.handles, self.events, self.active = [], [], None

    def __enter__(self):
        modules = [('embedding', 'embedding', self.model.embedding), ('final_norm', 'normalization', self.model.final_norm)]
        for index, layer in enumerate(self.model.layers):
            prefix = f'layers.{index}'
            modules.extend([(prefix + '.mixer', 'gdn_mixer' if layer.kind == 'gdn' else 'global_mixer', layer.mixer),
                            (prefix + '.ffn', 'ffn', layer.ffn), (prefix + '.input_norm', 'normalization', layer.input_norm),
                            (prefix + '.post_norm', 'normalization', layer.post_norm)])
        names = [name for name, _, _ in modules]
        require(not any(a != b and b.startswith(a + '.') for a in names for b in names), 'Nested timer modules forbidden')
        for name, group, module in modules:
            def before(_module, _args, name=name, group=group):
                require(self.active is None, 'Overlapping forward timer scopes')
                start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
                start.record()
                self.active = (name, group, start, end)
            def after(_module, _args, _output, name=name):
                require(self.active is not None and self.active[0] == name, 'Forward timer scope mismatch')
                self.active[3].record()
                self.events.append(self.active)
                self.active = None
            self.handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
        self.expected_names = names
        return self

    def results(self):
        require(self.active is None and sorted(x[0] for x in self.events) == sorted(self.expected_names), 'Each timed module must execute once')
        return [{'module': name, 'group': group, 'stream_elapsed_ms': float(start.elapsed_time(end))} for name, group, start, end in self.events]

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def execute(protocol, protocol_sha):
    import fcntl
    sys.path.insert(0, str(ROOT))
    import torch
    from scripts import run_babylm_checkpoint_eval_v0 as evaluator
    from src.babylm_hybrid.config import HybridConfig
    from src.babylm_hybrid.model import build_model
    from src.babylm_hybrid.local_attention_v0 import build_local_model, CONTRACT_BUFFER, LocalGlobalAttention
    from src.babylm_hybrid.training import _runtime_signature

    lock = open(protocol['project_lock'], 'a+b')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = resolve(protocol['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'protocol.json', protocol)
    begin, started = time.perf_counter(), utc()
    counts = {'forward_attempts': 0, 'forward_calls': 0, 'backward_attempts': 0, 'backward_calls': 0,
              'warmup_forward_attempts': 0, 'warmup_forward_calls': 0, 'optimizer_updates': 0, 'scientific_updates': 0,
              'submitted_input_tokens': 0, 'submitted_loss_tokens': 0, 'submitted_word_exposures': 0}
    summary = {'status': 'running', 'scope': protocol['scope'], 'started_utc': started, 'protocol_sha256': protocol_sha,
               'counts': counts, 'arms': {}, 'tolerance': TOLERANCE, 'automatic_retry': False,
               'timing_scope': 'CUDA current-stream elapsed and synchronized host wall; not pure kernel busy, shared-GPU diagnosis only.',
               'component_scope': 'Forward-only nonoverlapping module calls; never a fraction of training F+B. Residual includes tied functional lm_head, loss, masks, residuals and Python gaps.',
               'excluded_from_timed_region': ['zero_grad', 'input upload', 'CPU gradient comparison', 'weight hashing', 'model construction/loading', 'logging'],
               'backward_components': 'Not instrumented; only whole backward is measured.', 'no_optimizer_instantiated': True}
    summary['workload_boundary'] = 'One-window model forward/backward only; excludes optimizer, gradient accumulation/clipping, data loading and checkpointing. Stratified eight-window workload is not a corpus-average training cost.'
    stream = (output / 'events.jsonl').open('x', encoding='utf-8')

    def emit(kind, **fields):
        row = {'type': kind, 'utc': utc(), 'counts': dict(counts), **fields}
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())

    def budget():
        if time.perf_counter() - begin >= protocol['max_wall_seconds']:
            raise TimeoutError('Frozen profiling wall-time ceiling reached')

    def interrupted(_signum, _frame):
        raise InterruptedError('Profiling interrupted by outer supervisor')

    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in [signal.SIGTERM, signal.SIGINT]}
    try:
        source, dependency = evaluator.verify_sources(protocol)
        manifest, fingerprint = evaluator.verify_manifest(resolve(protocol['train_manifest']), protocol['train_manifest_sha256'])
        require(all(i < manifest['total_windows'] for i in protocol['window_indices']), 'Window outside fixed manifest')
        torch.set_num_threads(1); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        runtime = _runtime_signature('cuda')
        runtime.update({'device': 'cuda', 'dtype': 'float32'})
        require(runtime == protocol['expected_runtime'], 'Runtime differs from frozen execution contract')
        require(runtime['gpu']['uuid'].removeprefix('GPU-') == protocol['execution_hardware']['physical_uuid'].removeprefix('GPU-'), 'Runtime GPU differs from hardware contract')
        summary['shared_gpu_before'] = gpu_snapshot(protocol)
        free, total = torch.cuda.mem_get_info()
        require(free >= protocol['min_free_gpu_bytes'], 'Less than fixed 16 GiB free before model allocation')
        summary['gpu_memory_admission'] = {'free_bytes': free, 'total_bytes': total, 'minimum_free_bytes': protocol['min_free_gpu_bytes']}
        summary.update({'source_hashes': source, 'transformers_gdn_source': dependency, 'runtime': runtime, 'data_fingerprint': fingerprint})
        items = list(evaluator.window_reader.iter_windows(resolve(protocol['train_manifest']), protocol['window_indices'], verify_hashes=False))
        require([int(x['window_index']) for x in items] == protocol['window_indices'], 'Window order differs')
        require(all(x['single_segment'] and x['reset_model_state_before'] and int(x['loss_tokens']) > 0 for x in items), 'Nonempty independent windows required')
        by_id = {int(x['window_index']): x for x in items}
        for arm in ['D', 'W']:
            budget()
            cp = protocol['arms'][arm]
            cp_path = resolve(cp['checkpoint_path'])
            require(sha(cp_path) == cp['checkpoint_sha256'], 'Checkpoint file SHA differs: ' + arm)
            saved = torch.load(cp_path, map_location='cpu', weights_only=(cp.get('checkpoint_format', 'model_only') == 'model_only'))
            original = saved['protocol']
            require(saved['protocol_sha256'] == cp['checkpoint_protocol_sha256'] and canonical(original) == cp['checkpoint_protocol_sha256'], 'Checkpoint protocol binding differs')
            require(original['model_config'] == protocol['model_config'] and all(original[k] == protocol[k] for k in ['backbone_seed', 'indexer_seed']), 'Checkpoint model/seeds differ')
            require(saved['mode'] == 'dense', 'D/W frozen-engine compatibility mode expected')
            checkpoint_sources = list(evaluator.REQUIRED_SOURCES[1:])
            if arm == 'W':
                checkpoint_sources.append('src/babylm_hybrid/local_attention_v0.py')
            for name in checkpoint_sources:
                hits = [h for p, h in saved['source_hashes'].items() if p.replace('\\', '/').endswith(name)]
                require(hits and all(h == source[name] for h in hits), 'Checkpoint source mismatch: ' + name)
            dep_hits = [h for p, h in saved['source_hashes'].items() if p.replace('\\', '/').endswith('transformers/models/qwen3_next/modeling_qwen3_next.py')]
            require(dep_hits and all(h == dependency['sha256'] for h in dep_hits), 'Checkpoint GDN dependency mismatch')
            state = saved['model_state']
            require((CONTRACT_BUFFER in state) == (arm == 'W'), 'D/W checkpoint policy marker differs')
            for value in state.values():
                require(isinstance(value, torch.Tensor), 'Non-tensor model state')
                require(not value.is_floating_point() or value.dtype == torch.float32 and bool(torch.isfinite(value).all()), 'Invalid FP32 model state')
            factory = build_model if arm == 'D' else build_local_model
            model = factory(HybridConfig(**protocol['model_config']), 'dense', protocol['backbone_seed'], protocol['indexer_seed'])
            model.load_state_dict(state, strict=True)
            require(all(isinstance(layer.mixer, LocalGlobalAttention) == (arm == 'W') for layer in model.layers if layer.kind == 'global'), 'Wrong attention implementation')
            require(all(layer.mixer.indexer is None for layer in model.layers if layer.kind == 'global'), 'Indexer forbidden in D/W')
            before_hash = state_hash(torch, model)
            summary['arms'][arm] = {'checkpoint_sha256': cp['checkpoint_sha256'], 'checkpoint_protocol_sha256': cp['checkpoint_protocol_sha256'],
                                   'model_state_before_sha256': canonical(before_hash), 'parameter_tensors': len(list(model.named_parameters())), 'rows': []}
            del saved, state
            model.to('cuda', dtype=torch.float32).train()
            require(all(p.requires_grad for p in model.parameters()), 'All backbone parameters must participate in backward')
            budget()

            def run_pass(item, instrumented, warmup=False):
                budget(); model.zero_grad(set_to_none=True)
                require(counts['forward_attempts'] < MAX_F, 'Forward hard count reached')
                inputs = torch.tensor(item['input_ids'], dtype=torch.long, device='cuda').unsqueeze(0)
                require(inputs.shape[1] == item['input_tokens'] and item['loss_tokens'] == inputs.shape[1] - 1, 'Input accounting differs')
                identity = {'arm': arm, 'window_index': int(item['window_index']), 'source_index': int(item['source_index']), 'input_tokens': int(item['input_tokens']), 'loss_tokens': int(item['loss_tokens']), 'word_exposures': int(item['word_exposures']), 'instrumented': instrumented, 'warmup': warmup}
                counts['forward_attempts'] += 1
                counts['submitted_input_tokens'] += int(item['input_tokens']); counts['submitted_loss_tokens'] += int(item['loss_tokens'])
                counts['submitted_word_exposures'] += int(item['word_exposures'])
                if warmup: counts['warmup_forward_attempts'] += 1
                emit('forward_attempt', **identity)
                context = ForwardTimers(torch, model) if instrumented else nullcontext(None)
                torch.cuda.synchronize()
                fstart, fend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with context as timers, (torch.no_grad() if warmup else nullcontext()):
                    fwall = time.perf_counter(); fstart.record()
                    result = model(inputs, aux_weight=0.0)
                    fend.record(); torch.cuda.synchronize(); fwall = time.perf_counter() - fwall
                    counts['forward_calls'] += 1
                    if warmup: counts['warmup_forward_calls'] += 1
                    components = timers.results() if timers else []
                loss = float(result.lm_loss.detach().cpu())
                require(math.isfinite(loss) and result.token_loss_count == item['loss_tokens'], 'Nonfinite loss or wrong target count')
                forward_ms = float(fstart.elapsed_time(fend))
                emit('forward_complete', **identity, loss=loss, forward_stream_ms=forward_ms, forward_wall_seconds=fwall)
                budget()
                row = {**identity, 'loss': loss, 'forward_stream_ms': forward_ms, 'forward_wall_seconds': fwall, 'forward_components': components}
                if not warmup:
                    require(counts['backward_attempts'] < MAX_B, 'Backward hard count reached')
                    counts['backward_attempts'] += 1; emit('backward_attempt', **identity)
                    bstart, bend = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(); bwall = time.perf_counter(); bstart.record()
                    result.lm_loss.backward()
                    bend.record(); torch.cuda.synchronize(); bwall = time.perf_counter() - bwall
                    counts['backward_calls'] += 1
                    row.update({'backward_stream_ms': float(bstart.elapsed_time(bend)), 'backward_wall_seconds': bwall})
                    emit('backward_complete', **identity, backward_stream_ms=row['backward_stream_ms'], backward_wall_seconds=bwall)
                del result, inputs
                budget()
                return row

            for index in protocol.get('warmup_window_indices', []):
                warm = run_pass(by_id[index], False, True)
                emit('warmup_complete', row=warm)
            for position, item in enumerate(items):
                # Counterbalance serial-order drift while always comparing against bare.
                instrumented_first = position % 2 == 1
                first = run_pass(item, instrumented_first)
                first_grads = {name: None if param.grad is None else param.grad.detach().cpu().clone() for name, param in model.named_parameters()}
                second = run_pass(item, not instrumented_first)
                bare, instrumented = (second, first) if instrumented_first else (first, second)
                comparison = {'loss': compare_tensor(torch, torch.tensor(bare['loss']), torch.tensor(instrumented['loss'])), 'gradients': {}}
                for name, param in model.named_parameters():
                    reference, actual = (param.grad, first_grads[name]) if instrumented_first else (first_grads[name], param.grad)
                    require((reference is None) == (actual is None), 'Gradient participation changed: ' + name)
                    comparison['gradients'][name] = {'passed': True, 'both_none': True} if reference is None else compare_tensor(torch, reference, actual)
                comparison['passed'] = comparison['loss']['passed'] and all(x['passed'] for x in comparison['gradients'].values())
                emit('hook_equivalence', arm=arm, window_index=int(item['window_index']), comparison=comparison)
                require(comparison['passed'], 'Instrumentation changed loss or named gradients')
                groups = defaultdict(float)
                for part in instrumented['forward_components']:
                    groups[part['group']] += part['stream_elapsed_ms']
                total_components = sum(groups.values())
                # A tolerance of 0.01 ms covers timestamp arithmetic only, not numerical comparisons.
                require(total_components <= instrumented['forward_stream_ms'] + 0.01, 'Forward component events overlap or exceed whole forward')
                pair = {'window_index': int(item['window_index']), 'order': ['instrumented', 'bare'] if instrumented_first else ['bare', 'instrumented'], 'bare': bare, 'instrumented': instrumented,
                        'forward_group_stream_ms': dict(groups), 'forward_group_share': {k: v / instrumented['forward_stream_ms'] for k, v in groups.items()},
                        'unattributed_forward_stream_ms': instrumented['forward_stream_ms'] - total_components,
                        'hook_overhead_forward_stream_ms': instrumented['forward_stream_ms'] - bare['forward_stream_ms'],
                        'hook_overhead_forward_wall_seconds': instrumented['forward_wall_seconds'] - bare['forward_wall_seconds'],
                        'equivalence_passed': True}
                summary['arms'][arm]['rows'].append(pair)
                emit('paired_measurement_complete', arm=arm, row=pair)
                del first_grads
            model.zero_grad(set_to_none=True)
            after_hash = state_hash(torch, model)
            require(before_hash == after_hash and sha(cp_path) == cp['checkpoint_sha256'], 'Read-only weights mutated')
            info = summary['arms'][arm]
            info.update({'model_state_after_sha256': canonical(after_hash), 'weights_unchanged': True, 'checkpoint_file_unchanged': True})
            grouped = defaultdict(list)
            for row in info['rows']:
                for kind in ['bare', 'instrumented']:
                    for key in ['forward_stream_ms', 'backward_stream_ms', 'forward_wall_seconds', 'backward_wall_seconds']:
                        grouped[kind + '.' + key].append(row[kind][key])
                for group, value in row['forward_group_share'].items():
                    grouped['instrumented.forward_share.' + group].append(value)
                for group, value in row['forward_group_stream_ms'].items():
                    grouped['instrumented.forward_group_ms.' + group].append(value)
            info['timing_summary'] = {k: {'count': len(v), 'sum': sum(v), 'mean': statistics.mean(v), 'median': statistics.median(v), 'min': min(v), 'max': max(v)} for k, v in grouped.items()}
            denominator = sum(row['instrumented']['forward_stream_ms'] for row in info['rows'])
            info['pooled_instrumented_forward_share'] = {group: sum(row['forward_group_stream_ms'].get(group, 0.0) for row in info['rows']) / denominator for group in info['rows'][0]['forward_group_stream_ms']}
            info['share_aggregation_note'] = 'pooled_instrumented_forward_share = sum(component_ms)/sum(whole_instrumented_F_ms); per-window share distribution is descriptive, not corpus weighted or a training F+B share.'
            emit('arm_complete', arm=arm, state_before_sha256=canonical(before_hash), state_after_sha256=canonical(after_hash))
            del model
            torch.cuda.empty_cache()
            budget()
        require(counts['forward_attempts'] == counts['forward_calls'] == protocol['expected_calls']['forward_attempts'] and counts['backward_attempts'] == counts['backward_calls'] == protocol['expected_calls']['backward_attempts'], 'Final workload differs')
        summary['shared_gpu_after'] = gpu_snapshot(protocol)
        budget()
        summary['status'] = 'complete_read_only_cost_diagnostic'
        return_code = 0
    except BaseException as exc:
        summary.update({'status': 'failed_or_incomplete', 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
        emit('failure', error_type=type(exc).__name__, error=str(exc))
        return_code = 1
    finally:
        summary.update({'finished_utc': utc(), 'elapsed_wall_seconds': time.perf_counter() - begin})
        stream.close()
        summary['events_sha256'] = sha(output / 'events.jsonl')
        write(output / 'summary.json', summary)
        for sig, handler in previous_handlers.items(): signal.signal(sig, handler)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()
    return return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--protocol-sha256', '--protocolsha', dest='protocol_sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    path = resolve(args.protocol)
    require(sha(path) == args.protocol_sha256, 'Protocol file SHA differs')
    protocol = json.loads(path.read_text(encoding='utf-8-sig'))
    expected = validate(protocol)
    if not args.execute:
        print(json.dumps({'status': 'validated_no_model_calls', 'expected_calls': expected, 'model_calls': 0, 'cuda_calls': 0}))
        return 0
    require(protocol.get('launch_allowed') is True, 'Execution not authorized by frozen protocol')
    return execute(protocol, args.protocol_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
