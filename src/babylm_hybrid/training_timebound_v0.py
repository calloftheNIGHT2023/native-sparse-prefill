"""Administrative time-bounded copy of frozen training.py; no scientific changes.

Unknown CUDA prices are represented by null, never an invented monetary rate.
The explicit time_bounded_unknown_rate mode requires a finite positive wall
limit. The caller must still enforce an external hard process-group deadline.

Checkpoints are valid at committed update/data boundaries. A Python exception
saves a separate non-resumable failure snapshot, never disguising a partially
executed step as an exact replay point. Time checks are cooperative boundaries,
not an external hard timeout. The current model is a correctness reference.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import random
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .config import HybridConfig
from .model import build_model, parameter_counts, Qwen3NextGatedDeltaNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_TOKEN_ROOT = PROJECT_ROOT / 'data/babylm-2026-tokenizer-16k-v0'
WINDOW_COLUMNS = ['source_index', 'segment_index_in_source', 'source_token_start',
                  'source_token_end', 'first_record_index', 'last_record_index',
                  'first_record_token_start', 'last_record_token_end',
                  'word_exposures', 'input_tokens', 'next_token_loss_positions']


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def protocol_sha256(protocol):
    return hashlib.sha256(_canonical(protocol)).hexdigest()


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name('config.py'),
             Path(__file__).with_name('model.py'), Path(__file__).with_name('attention.py'),
             Path(inspect.getfile(Qwen3NextGatedDeltaNet)),
             PROJECT_ROOT / 'scripts/prepare_babylm_windows_v0.py',
             PROJECT_ROOT / 'scripts/run_babylm_de_v0.py']
    evaluation = Path(__file__).with_name('evaluation.py')
    if evaluation.exists():
        paths.append(evaluation)
    return {str(path.resolve()): _file_hash(path) for path in paths}


def _protocol_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _runtime_signature(device):
    gpu = None
    if device == 'cuda':
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        identifier = getattr(properties, 'uuid', None)
        gpu = {'device_index': torch.cuda.current_device(), 'name': properties.name,
               'capability': [properties.major, properties.minor],
               'total_memory_bytes': properties.total_memory,
               'uuid': str(identifier) if identifier is not None else None}
    return {'torch': torch.__version__, 'numpy': np.__version__,
            'torch_num_threads': torch.get_num_threads(), 'cuda_version': torch.version.cuda,
            'gpu': gpu, 'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}


def word_lr_multiplier(words, protocol):
    """Shared backbone/indexer multiplier, evaluated at update-end word exposure."""
    cap = int(protocol['max_word_exposures'])
    warmup = int(protocol['warmup_word_exposures'])
    minimum = float(protocol['min_lr_ratio'])
    if warmup and words < warmup:
        return max(0.0, float(words) / warmup)
    progress = min(1.0, max(0.0, (float(words) - warmup) / max(cap - warmup, 1)))
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def epoch_permutation(seed, epoch, length):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)])).permutation(length)


class _WindowDataset:
    def __init__(self, protocol):
        path = _protocol_path(protocol['train_manifest'])
        actual_sha = _file_hash(path)
        if actual_sha != protocol['train_manifest_sha256']:
            raise ValueError('Train manifest SHA256 mismatch')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('status') != 'window_index_word_accounting_and_coverage_verified' or manifest.get('columns') != WINDOW_COLUMNS:
            raise ValueError('Unsupported frozen training-window manifest')
        artifacts = {}
        for artifact in manifest['artifacts']:
            item = (path.parent / artifact['path']).resolve()
            if item.parent != path.parent or _file_hash(item) != artifact['sha256']:
                raise ValueError('Training window artifact path/hash mismatch')
            artifacts[str(item)] = artifact['sha256']
        self.index = np.load(path.parent / 'windows.u64.npy', mmap_mode='r', allow_pickle=False)
        if self.index.shape != (manifest['total_windows'], len(WINDOW_COLUMNS)):
            raise ValueError('Training window index shape mismatch')
        self.streams = {}
        for source, entry in enumerate(manifest['source_summaries']):
            tokens = (PROJECT_ROOT / entry['token_ids_path_relative_to_project']).resolve()
            if not tokens.is_relative_to(TRAIN_TOKEN_ROOT.resolve()) or not tokens.name.endswith('.ids.u32'):
                raise ValueError('Training token stream is outside the frozen training ledger')
            digest = _file_hash(tokens)
            if digest != entry['token_ids_sha256']:
                raise ValueError('Training token stream SHA256 mismatch')
            artifacts[str(tokens)] = digest
            self.streams[source] = np.memmap(tokens, mode='r', dtype='<u4')
        self.fingerprint = {'manifest_path': str(path), 'manifest_sha256': actual_sha, 'artifacts': artifacts,
                            'total_windows': len(self.index), 'tokenizer_sha256': manifest['tokenizer_sha256']}

    def __len__(self):
        return len(self.index)

    def window(self, index):
        row = [int(value) for value in self.index[int(index)]]
        source, segment, start, end, first, last, left, right, words, tokens, targets = row
        view = self.streams[source][start:end]
        if len(view) != tokens or targets != max(tokens - 1, 0):
            raise ValueError('Window accounting is inconsistent with its single reset segment')
        return {'window_index': int(index), 'input_ids': view, 'word_exposures': words,
                'input_tokens': tokens, 'next_token_loss_positions': targets,
                'source_index': source, 'segment_index_in_source': segment}


def load_window_dataset(protocol):
    """Single test injection boundary; production never accepts user-supplied loaders."""
    return _WindowDataset(protocol)


def _validate_protocol(p):
    fields = ['schema_version', 'scope', 'launch_allowed', 'device', 'dtype', 'model_config',
              'backbone_seed', 'indexer_seed', 'data_order_seed', 'train_manifest',
              'train_manifest_sha256', 'max_word_exposures', 'max_updates', 'max_wall_seconds',
              'windows_per_update', 'learning_rate', 'indexer_learning_rate', 'weight_decay',
              'betas', 'eps', 'grad_clip_norm', 'aux_weight', 'warmup_word_exposures',
              'min_lr_ratio', 'checkpoint_every_updates', 'eval_every_updates']
    missing = [key for key in fields if key not in p]
    if missing:
        raise ValueError(f'Missing protocol fields: {missing}')
    if p['schema_version'] != 1 or p['scope'] not in ('engineering_smoke', 'gpu_preflight', 'scientific'):
        raise ValueError('Unsupported protocol schema/scope')
    if p['launch_allowed'] is not True:
        raise ValueError('Protocol launch_allowed must be explicitly true')
    if p['dtype'] != 'float32' or p['device'] not in ('cpu', 'cuda'):
        raise ValueError('This engine supports only explicit CPU/CUDA float32')
    for key in ('max_word_exposures', 'max_updates', 'windows_per_update', 'checkpoint_every_updates'):
        if type(p[key]) is not int or p[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if 'max_epochs' in p and (type(p['max_epochs']) is not int or p['max_epochs'] <= 0):
        raise ValueError('max_epochs must be a positive integer when specified')
    if p.get('gradient_clip_scope', 'global') not in ('global', 'separate_backbone_indexer'):
        raise ValueError('Unsupported gradient_clip_scope')
    for key in ('backbone_seed', 'indexer_seed', 'data_order_seed', 'warmup_word_exposures', 'eval_every_updates'):
        if type(p[key]) is not int or p[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if p['warmup_word_exposures'] >= p['max_word_exposures']:
        raise ValueError('Warmup must end before the fixed word cap')
    for key in ('learning_rate', 'indexer_learning_rate', 'eps', 'grad_clip_norm', 'max_wall_seconds'):
        if not math.isfinite(float(p[key])) or p[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('weight_decay', 'aux_weight'):
        if not math.isfinite(float(p[key])) or p[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if not 0 <= p['min_lr_ratio'] <= 1 or len(p['betas']) != 2 or any(not 0 <= beta < 1 for beta in p['betas']):
        raise ValueError('Invalid LR floor or AdamW betas')
    if not isinstance(p['train_manifest_sha256'], str) or len(p['train_manifest_sha256']) != 64:
        raise ValueError('A frozen train manifest SHA256 is required')
    milestones = p.get('milestone_word_exposures', [])
    if (not isinstance(milestones, list) or
            any(type(value) is not int or value <= 0 for value in milestones) or
            milestones != sorted(set(milestones))):
        raise ValueError('milestone_word_exposures must be a strictly increasing list of positive integers')
    unknown_rate = p.get('cost_mode') == 'time_bounded_unknown_rate'
    if unknown_rate:
        if p['device'] != 'cuda' or p['scope'] not in ('scientific', 'gpu_preflight'):
            raise ValueError('Unknown-rate time bounds require an explicit CUDA stage')
        if ('hourly_rate_usd' not in p or p['hourly_rate_usd'] is not None or
                'paid_ceiling_usd' not in p or p['paid_ceiling_usd'] is not None):
            raise ValueError('Unknown-rate mode requires explicit null hourly_rate_usd and paid_ceiling_usd')
        if (type(p['max_wall_seconds']) not in (int, float) or
                not math.isfinite(p['max_wall_seconds']) or p['max_wall_seconds'] <= 0):
            raise ValueError('Unknown-rate mode requires a finite positive wall-time bound')
    if p['scope'] == 'engineering_smoke':
        if p['device'] != 'cpu' or p['max_updates'] > 10:
            raise ValueError('Engineering smoke requires CPU and at most 10 updates')
    else:
        if not unknown_rate:
            ceiling = float(p.get('paid_ceiling_usd', 0))
            spent = float(p.get('stage_spent_usd', 0))
            if not math.isfinite(ceiling) or ceiling <= 0 or not 0 <= spent < ceiling:
                raise ValueError('Paid stage requires a positive unexhausted paid_ceiling_usd')
        if p['scope'] == 'gpu_preflight' and (p['max_updates'] > 12 or p['max_wall_seconds'] > 1800):
            raise ValueError('GPU preflight is limited to 12 updates and 1800 seconds')
    if p['device'] == 'cuda' and not unknown_rate and (not math.isfinite(float(p.get('hourly_rate_usd', 0))) or p.get('hourly_rate_usd', 0) <= 0):
        raise ValueError('CUDA execution requires hourly_rate_usd for the stage cost bound')
    if p.get('eval_manifest'):
        if not p.get('eval_manifest_sha256') or _file_hash(_protocol_path(p['eval_manifest'])) != p['eval_manifest_sha256']:
            raise ValueError('Evaluation manifest SHA256 mismatch')
        if p.get('eval_window_indices') is not None and (not isinstance(p['eval_window_indices'], list) or
                any(type(index) is not int or index < 0 for index in p['eval_window_indices'])):
            raise ValueError('eval_window_indices must be an explicit list of nonnegative integers')
    elif p['eval_every_updates'] or p.get('eval_initial', False) or p.get('eval_final', False):
        raise ValueError('Evaluation cadence requires an evaluation manifest')
    for name in ('eval_initial', 'eval_final', 'eval_position_diagnostics'):
        if name in p and type(p[name]) is not bool:
            raise ValueError(f'{name} must be a boolean')
    if p.get('eval_manifest') and (p['eval_every_updates'] or p.get('eval_initial') or p.get('eval_final')):
        if not p.get('eval_window_indices'):
            raise ValueError('A nonempty frozen eval_window_indices panel is required; no silent full-dev evaluation')


def _atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('wb') as stream:
        stream.write(_canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_torch(path, value):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _immutable_save(path, value, *, torch_format=False):
    """Publish complete evidence once; never replace an existing destination.

    A same-directory hard link is an atomic create-if-absent operation on the
    supported local filesystems. On failure the temporary evidence is retained
    and the error propagates to the normal failure logger. No fallback overwrites
    an existing snapshot, and an unsupported filesystem fails closed.
    """
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        if torch_format:
            torch.save(value, stream)
        else:
            stream.write(_canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _gradient_l2_norm(parameters):
    """Read the accumulated combined-objective gradients without altering them."""
    with torch.no_grad():
        norms = [torch.linalg.vector_norm(parameter.grad.detach(), 2)
                 for parameter in parameters if parameter.grad is not None]
        return float(torch.linalg.vector_norm(torch.stack(norms), 2)) if norms else 0.0


def _clip_parameter_groups(model, main_params, index_params, max_norm, scope):
    """Keep legacy clipping unchanged unless independent groups are explicit."""
    def coefficient(norm):
        return float(torch.clamp(max_norm / (norm.detach() + 1e-6), max=1.0))

    if scope == 'global':
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
        shared = coefficient(norm)
        return norm, shared, shared, shared if index_params else 1.0
    if scope != 'separate_backbone_indexer':
        raise ValueError('Unsupported gradient_clip_scope')
    main_norm = torch.nn.utils.clip_grad_norm_(main_params, max_norm, error_if_nonfinite=True)
    if index_params:
        index_norm = torch.nn.utils.clip_grad_norm_(index_params, max_norm, error_if_nonfinite=True)
        index_coefficient = coefficient(index_norm)
    else:
        index_norm = main_norm.new_zeros(())
        index_coefficient = 1.0
    # clip_grad_norm_ returns each group's norm BEFORE modifying that group.
    total_norm = torch.linalg.vector_norm(torch.stack((main_norm, index_norm)), 2)
    main_coefficient = coefficient(main_norm)
    return total_norm, main_coefficient, main_coefficient, index_coefficient


def _advance(cursor, length):
    cursor = dict(cursor)
    cursor['position'] += 1
    if cursor['position'] == length:
        cursor = {'epoch': cursor['epoch'] + 1, 'position': 0}
    return cursor


def train_run(protocol: dict, mode: str, output_dir: Path, resume_checkpoint: Path | None = None,
              stop_after_updates: int | None = None):
    """Run a frozen D/E baseline; stop_after_updates is an absolute update count.

    Resume is allowed only in the same output directory, from a checkpoint whose
    complete log prefix is still the entire current log. Divergent/extra events
    require an explicit recovery audit, never silent truncation or reexecution.
    """
    started_clock = time.perf_counter()
    p = copy.deepcopy(protocol)
    _validate_protocol(p)
    if mode not in ('dense', 'sparse'):
        raise ValueError('Only dense/sparse baseline modes are supported')
    if stop_after_updates is not None and (type(stop_after_updates) is not int or stop_after_updates < 0):
        raise ValueError('stop_after_updates must be an absolute nonnegative integer')
    run_dir = Path(output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / 'run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'utc': _utc(), 'mode': mode}, stream)
    log_path, checkpoint_path = run_dir / 'events.jsonl', run_dir / 'checkpoint.pt'
    phash, shashes = protocol_sha256(p), source_hashes()
    event_id = 0
    model = optimizer = dataset = None
    pending = []
    counts = dict(word_exposures=0, input_tokens=0, loss_tokens=0, forward_calls=0,
                  backward_calls=0, forward_attempts=0, backward_attempts=0, updates=0,
                  windows=0, scientific_updates=0, engineering_updates=0,
                  skipped_zero_target_input_tokens=0, skipped_zero_target_word_exposures=0,
                  forward_input_tokens=0, forward_word_exposures=0)
    eval_counts = dict(word_exposures=0, input_tokens=0, loss_tokens=0, forward_calls=0, evaluations=0)
    cursor = {'epoch': 0, 'position': 0}
    elapsed_before = 0.0
    last_lr = {}
    last_eval_point = None
    active_evaluation = None
    initial_parameter_hashes = {}
    snapshot_state = {'models': [], 'crossed_milestones': [], 'final_receipts': []}
    state_ready = False

    def elapsed():
        return elapsed_before + time.perf_counter() - started_clock

    def estimated_paid_cost():
        if p.get('cost_mode') == 'time_bounded_unknown_rate':
            return None
        return elapsed() / 3600 * float(p.get('hourly_rate_usd', 0)) if p['device'] == 'cuda' else 0.0

    def append(kind, **payload):
        nonlocal event_id
        event_id += 1
        value = {'event_id': event_id, 'type': kind, 'utc': _utc(), 'mode': mode, **payload}
        if p.get('cost_mode') == 'time_bounded_unknown_rate':
            value.update(cost_mode='time_bounded_unknown_rate', hourly_rate_usd=None, paid_ceiling_usd=None)
        with log_path.open('ab') as stream:
            stream.write(_canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
        return value

    def checkpoint(destination=checkpoint_path, resumable=True, reason=None):
        value = {'checkpoint_schema_version': 1, 'resume_supported': resumable, 'reason': reason,
                 'run_dir': str(run_dir), 'mode': mode, 'protocol': p, 'protocol_sha256': phash,
                 'source_hashes': shashes, 'data_fingerprint': dataset.fingerprint,
                 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                 'counters': dict(counts), 'counts': dict(counts), 'eval_counts': dict(eval_counts),
                 'cursor': dict(cursor), 'pending_batch': copy.deepcopy(pending), 'last_lr': dict(last_lr),
                 'last_eval_point': copy.deepcopy(last_eval_point),
                 'active_evaluation': copy.deepcopy(active_evaluation),
                 'initial_parameter_hashes': initial_parameter_hashes,
                 'snapshot_state': copy.deepcopy(snapshot_state),
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if p['device'] == 'cuda' else None,
                 'elapsed_wall_seconds': elapsed(), 'event_id': event_id,
                 'log_bytes': log_path.stat().st_size, 'log_sha256': _file_hash(log_path),
                 'runtime': _runtime_signature(p['device']), 'saved_utc': _utc()}
        _atomic_torch(destination, value)

    def snapshot_point():
        # These are actual ledger counters, never rounded nominal thresholds.
        return {key: counts[key] for key in ('updates', 'word_exposures', 'input_tokens', 'loss_tokens')}

    def snapshot_stem(point):
        return 'model-u{updates:08d}-w{word_exposures:012d}-i{input_tokens:012d}-l{loss_tokens:012d}'.format(**point)

    def model_snapshot(crossed):
        point = snapshot_point()
        for entry in snapshot_state['models']:
            if entry['point'] == point:
                return entry
        directory = run_dir / 'snapshots'
        directory.mkdir(exist_ok=True)
        path = directory / (snapshot_stem(point) + '.pt')
        provenance = {'snapshot_schema_version': 1, 'kind': 'model_only', 'resume_supported': False,
                      'point': point, 'counts': dict(counts), 'cursor': dict(cursor), 'mode': mode,
                      'nominal_crossed_milestones': list(crossed), 'protocol_sha256': phash,
                      'source_hashes': shashes, 'data_fingerprint': dataset.fingerprint,
                      'initial_parameter_hashes': initial_parameter_hashes,
                      'saved_utc': _utc(),
                      'word_accounting': 'actual consumed words; forward_word_exposures and skipped_zero_target_word_exposures are separately recorded in counts',
                      'purpose': 'immutable model evidence; branching requires a separately frozen protocol; no optimizer or RNG state'}
        _immutable_save(path, {**provenance, 'protocol': p, 'model_state': model.state_dict()}, torch_format=True)
        metadata_path = path.with_suffix('.json')
        entry = {**provenance, 'path': str(path.relative_to(run_dir)), 'sha256': _file_hash(path)}
        _immutable_save(metadata_path, entry)
        entry = {**entry, 'metadata_path': str(metadata_path.relative_to(run_dir)),
                 'metadata_sha256': _file_hash(metadata_path)}
        snapshot_state['models'].append(entry)
        return entry

    def record_milestones():
        crossed = [threshold for threshold in p.get('milestone_word_exposures', [])
                   if threshold <= counts['word_exposures'] and threshold not in snapshot_state['crossed_milestones']]
        if not crossed:
            return False
        entry = model_snapshot(crossed)
        snapshot_state['crossed_milestones'].extend(crossed)
        append('milestone_snapshot', nominal_crossed_milestones=crossed,
               snapshot_path=entry['path'], snapshot_sha256=entry['sha256'],
               point=snapshot_point(), counts=dict(counts), cursor=dict(cursor))
        return True

    def record_final_snapshot(stop_reason):
        point = snapshot_point()
        if any(item['point'] == point and item['stop_reason'] == stop_reason
               for item in snapshot_state['final_receipts']):
            return
        entry = model_snapshot([])
        receipt_path = run_dir / 'snapshots' / (snapshot_stem(point) + '-final-' + stop_reason + '.json')
        receipt = {'snapshot_schema_version': 1, 'kind': 'final_for_this_invocation',
                   'point': point, 'counts': dict(counts), 'cursor': dict(cursor), 'stop_reason': stop_reason,
                   'protocol_word_cap': p['max_word_exposures'],
                   'remaining_nominal_word_budget': p['max_word_exposures'] - counts['word_exposures'],
                   'nominal_milestones_reached': list(snapshot_state['crossed_milestones']),
                    'protocol_completion_boundary': (stop_reason == 'epoch_complete' if 'max_epochs' in p
                                                     else stop_reason in ('word_budget_reached', 'max_updates_reached')),
                   'model_path': entry['path'], 'model_sha256': entry['sha256'],
                   'protocol_sha256': phash, 'source_hashes': shashes, 'saved_utc': _utc()}
        _immutable_save(receipt_path, receipt)
        snapshot_state['final_receipts'].append({**receipt, 'path': str(receipt_path.relative_to(run_dir)),
                                                 'sha256': _file_hash(receipt_path)})
        append('final_model_snapshot', receipt_path=str(receipt_path.relative_to(run_dir)),
               snapshot_path=entry['path'], snapshot_sha256=entry['sha256'],
               stop_reason=stop_reason, point=point, counts=dict(counts), cursor=dict(cursor))

    def verify_snapshot_state(saved_state, committed_counts):
        if not isinstance(saved_state, dict) or set(saved_state) != {'models', 'crossed_milestones', 'final_receipts'}:
            raise ValueError('Resume snapshot state missing or malformed')
        if any(not isinstance(saved_state[key], list) for key in saved_state):
            raise ValueError('Resume snapshot lists malformed')
        seen_points = set()
        for item in saved_state['models']:
            if (set(item['point']) != {'updates', 'word_exposures', 'input_tokens', 'loss_tokens'} or
                    any(type(value) is not int or value < 0 or value > committed_counts[key]
                        or value != item['counts'][key] for key, value in item['point'].items())):
                raise ValueError('Resume model snapshot actual counters mismatch')
            if any(threshold > item['point']['word_exposures'] for threshold in item['nominal_crossed_milestones']):
                raise ValueError('Resume snapshot claims an unreached nominal milestone')
            point_key = _canonical(item['point'])
            if point_key in seen_points:
                raise ValueError('Resume duplicated model snapshot point')
            seen_points.add(point_key)
            for path_key, hash_key in (('path', 'sha256'), ('metadata_path', 'metadata_sha256')):
                path = (run_dir / item[path_key]).resolve()
                if (path.parent != (run_dir / 'snapshots').resolve() or not path.is_file() or
                        _file_hash(path) != item[hash_key]):
                    raise ValueError('Resume model snapshot evidence path/hash mismatch')
            if item['protocol_sha256'] != phash or item['source_hashes'] != shashes:
                raise ValueError('Resume model snapshot provenance mismatch')
        for item in saved_state['final_receipts']:
            path = (run_dir / item['path']).resolve()
            if (path.parent != (run_dir / 'snapshots').resolve() or not path.is_file() or
                    _file_hash(path) != item['sha256']):
                raise ValueError('Resume final snapshot receipt path/hash mismatch')
            if not any(model['path'] == item['model_path'] and model['sha256'] == item['model_sha256']
                       and model['point'] == item['point'] for model in saved_state['models']):
                raise ValueError('Resume final receipt has no matching model snapshot')
        reached = saved_state['crossed_milestones']
        recorded = [threshold for item in saved_state['models'] for threshold in item['nominal_crossed_milestones']]
        if reached != sorted(set(reached)) or reached != recorded:
            raise ValueError('Resume milestone deduplication state mismatch')
        if any(threshold not in p.get('milestone_word_exposures', []) for threshold in reached):
            raise ValueError('Resume snapshot milestone not in the frozen protocol')
        expected = [threshold for threshold in p.get('milestone_word_exposures', [])
                    if threshold <= committed_counts['word_exposures']]
        if reached != expected:
            raise ValueError('Resume snapshot state omitted a reached milestone')
        return copy.deepcopy(saved_state)

    def cost_time_reason():
        if elapsed() >= p['max_wall_seconds']:
            return 'max_wall_seconds_reached'
        if (p.get('cost_mode') != 'time_bounded_unknown_rate' and p['scope'] != 'engineering_smoke'
                and float(p.get('stage_spent_usd', 0)) + estimated_paid_cost() >= float(p['paid_ceiling_usd'])):
            return 'paid_ceiling_reached'
        return None

    def boundary_reason():
        budget_reason = cost_time_reason()
        if budget_reason:
            return budget_reason
        if 'max_epochs' in p and cursor['epoch'] >= p['max_epochs']:
            return 'epoch_complete'
        if counts['updates'] >= p['max_updates']:
            return 'max_updates_reached'
        if stop_after_updates is not None and counts['updates'] >= stop_after_updates:
            return 'stopped_by_update_limit'
        return None

    def evaluate(stage):
        nonlocal last_eval_point, active_evaluation
        point = {'updates': counts['updates'], 'word_exposures': counts['word_exposures']}
        if point == last_eval_point:
            return
        reason = cost_time_reason()
        if reason:
            append('eval_skipped_budget', stage=stage, reason=reason, counts=dict(counts), cursor=dict(cursor))
            return
        from .evaluation import evaluate_windows
        active_evaluation = {'stage': stage, 'point': point, 'started_utc': _utc(),
                             'partial_forward_count_unknown_if_evaluator_raises': True}
        evaluation_options = {'position_diagnostics': True} if p.get('eval_position_diagnostics', False) else {}
        metrics = evaluate_windows(model, _protocol_path(p['eval_manifest']), p['eval_window_indices'],
                                   device=p['device'], **evaluation_options)
        metric_totals = metrics.get('total', metrics)
        for key in ('word_exposures', 'input_tokens', 'loss_tokens', 'forward_calls'):
            if key not in metric_totals:
                raise ValueError(f'Evaluation result omitted required accounting field: {key}')
            eval_counts[key] += int(metric_totals[key])
        eval_counts['evaluations'] += 1
        last_eval_point = point
        active_evaluation = None
        append('evaluation', stage=stage, metrics=metrics, eval_counts=dict(eval_counts),
               counts=dict(counts), cursor=dict(cursor), elapsed_wall_seconds=elapsed())
        model.train()
        checkpoint(reason=f'evaluation_{stage}_committed')

    try:
        if resume_checkpoint is None and (log_path.exists() or checkpoint_path.exists() or (run_dir / 'protocol.json').exists()):
            raise ValueError('Existing run evidence requires explicit resume; refusing overwrite')
        dataset = load_window_dataset(p)
        if not len(dataset):
            raise ValueError('No training windows')
        cfg_dict = dict(p['model_config'])
        cfg_dict['layer_types'] = tuple(cfg_dict['layer_types'])
        cfg = HybridConfig(**cfg_dict)
        model = build_model(cfg, mode, p['backbone_seed'], p['indexer_seed'])
        sizes = parameter_counts(model)
        initial_parameter_hashes = {
            name: hashlib.sha256(_canonical({'shape': list(parameter.shape), 'dtype': str(parameter.dtype)})
                                 + parameter.detach().contiguous().numpy().tobytes()).hexdigest()
            for name, parameter in model.named_parameters()}
        if p['scope'] == 'engineering_smoke' and sizes['total'] > 1_000_000:
            raise ValueError('Engineering smoke model exceeds one million parameters')
        if p['device'] == 'cuda' and not torch.cuda.is_available():
            raise ValueError('Requested CUDA is unavailable; refusing a silent CPU fallback')
        device = torch.device(p['device'])
        model = model.to(device=device, dtype=torch.float32)
        main_params, index_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                raise ValueError(f'Unexpected frozen baseline parameter: {name}')
            (index_params if '.indexer.' in name else main_params).append(parameter)
        groups = [{'params': main_params, 'lr': p['learning_rate'], 'name': 'backbone'}]
        if index_params:
            groups.append({'params': index_params, 'lr': p['indexer_learning_rate'], 'name': 'indexer'})
        optimizer = torch.optim.AdamW(groups, weight_decay=p['weight_decay'], betas=tuple(p['betas']), eps=p['eps'])
        if resume_checkpoint is not None:
            resume = Path(resume_checkpoint).resolve()
            if resume.parent != run_dir:
                raise ValueError('Resume checkpoint must be in the same run directory')
            saved = torch.load(resume, map_location='cpu', weights_only=False)
            if not saved.get('resume_supported') or saved.get('checkpoint_schema_version') != 1:
                raise ValueError('Checkpoint is not a committed replay point')
            for key, expected in [('run_dir', str(run_dir)), ('mode', mode), ('protocol_sha256', phash),
                                  ('source_hashes', shashes), ('data_fingerprint', dataset.fingerprint)]:
                if saved.get(key) != expected:
                    raise ValueError(f'Resume {key} mismatch')
            if not log_path.exists() or log_path.stat().st_size != saved['log_bytes'] or _file_hash(log_path) != saved['log_sha256']:
                raise ValueError('Resume log bytes/hash mismatch; refusing event loss or duplicated work')
            events = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()]
            if len(events) != saved['event_id'] or any(row['event_id'] != i + 1 for i, row in enumerate(events)):
                raise ValueError('Resume event sequence mismatch')
            if events and events[-1].get('counts', saved['counters']) != saved['counters']:
                raise ValueError('Resume checkpoint/log counters mismatch')
            if saved['runtime'] != _runtime_signature(p['device']):
                raise ValueError('Resume runtime mismatch requires a separate portability audit')
            restored_snapshots = verify_snapshot_state(saved.get('snapshot_state'), saved['counters'])
            model.load_state_dict(saved['model_state'], strict=True)
            optimizer.load_state_dict(saved['optimizer_state'])
            counts.update(saved['counters']); eval_counts.update(saved['eval_counts'])
            cursor.update(saved['cursor']); last_lr.update(saved['last_lr'])
            last_eval_point = saved.get('last_eval_point')
            snapshot_state = restored_snapshots
            if saved.get('initial_parameter_hashes') != initial_parameter_hashes:
                raise ValueError('Resume random-initialization provenance mismatch')
            event_id = saved['event_id']; elapsed_before = float(saved['elapsed_wall_seconds'])
            summary_path = run_dir / 'summary.json'
            if summary_path.exists():
                previous = json.loads(summary_path.read_text(encoding='utf-8'))
                if previous.get('checkpoint_sha256') == _file_hash(resume) and previous.get('counts') == counts:
                    elapsed_before = max(elapsed_before, float(previous['elapsed_wall_seconds']))
            random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng']); torch.set_rng_state(saved['torch_rng'])
            if p['device'] == 'cuda':
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            state_ready = True
            append('resume', counts=dict(counts), cursor=dict(cursor), checkpoint_sha256=_file_hash(resume))
        else:
            random.seed(p['backbone_seed']); np.random.seed(p['data_order_seed'] % (2**32)); torch.manual_seed(p['backbone_seed'])
            if p['device'] == 'cuda':
                torch.cuda.manual_seed_all(p['backbone_seed'])
            _atomic_json(run_dir / 'protocol.json', p)
            state_ready = True
            append('run_start', protocol_sha256=phash, source_hashes=shashes, data_fingerprint=dataset.fingerprint,
                   parameter_counts=sizes, counts=dict(counts), cursor=dict(cursor), scope=p['scope'],
                   initial_parameter_hashes=initial_parameter_hashes,
                   checkpoint_retention='latest optimizer checkpoint; immutable milestone/final model snapshots; unique failure snapshots',
                   time_limit_kind='cooperative_step_boundaries_not_external_hard_timeout')
            checkpoint(reason='initialized')
        model.train()
        if p.get('eval_initial', False) and counts['updates'] == 0 and last_eval_point is None:
            evaluate('initial')
        stop_reason = None
        zero_target_windows_since_update = 0
        while stop_reason is None:
            stop_reason = boundary_reason()
            if stop_reason:
                break
            pending = []
            planned = []
            planned_cursor = dict(cursor)
            planned_words = counts['word_exposures']
            cap_pending = False
            epoch_pending = False
            for _ in range(p['windows_per_update']):
                # Do not even read a window or generate a permutation from the
                # next epoch. A short final accumulation is committed normally.
                if 'max_epochs' in p and planned_cursor['epoch'] >= p['max_epochs']:
                    epoch_pending = True
                    break
                if boundary_reason():
                    stop_reason = boundary_reason()
                    break
                order = epoch_permutation(p['data_order_seed'], planned_cursor['epoch'], len(dataset))
                window_id = int(order[planned_cursor['position']])
                item = dataset.window(window_id)
                tokens = np.asarray(item['input_ids'])
                words, length, targets = (int(item[key]) for key in ('word_exposures', 'input_tokens', 'next_token_loss_positions'))
                if tokens.ndim != 1 or len(tokens) != length or length < 1 or targets != max(length - 1, 0) or words < 0:
                    raise ValueError('Invalid single-segment window token/word/target accounting')
                if bool((tokens < 0).any()) or bool((tokens >= cfg.vocab_size).any()):
                    raise ValueError('Window token IDs exceed the frozen model vocabulary')
                if planned_words + words > p['max_word_exposures']:
                    cap_pending = True
                    break  # Never skip an expensive window to consume a later cheaper one.
                identifier = {'epoch': planned_cursor['epoch'], 'window_index': window_id}
                after = _advance(planned_cursor, len(dataset))
                planned.append((item, identifier, after))
                pending.append({**identifier, 'word_exposures': words, 'input_tokens': length,
                                'loss_tokens': targets, 'forward_started': False, 'forward_completed': False,
                                'backward_started': False, 'backward_completed': False})
                planned_words += words
                planned_cursor = after
            if stop_reason:
                pending = []  # No forward/data exposure occurred for this merely-read plan.
                break
            if not planned:
                stop_reason = ('epoch_complete' if epoch_pending else
                               'word_budget_reached' if cap_pending else 'no_windows_planned')
                break
            targets_total = sum(int(item['next_token_loss_positions']) for item, _, _ in planned)
            optimizer.zero_grad(set_to_none=True)
            update_start = time.perf_counter()
            ce_sum = auxiliary_sum = 0.0
            attention_totals = {}
            for ordinal, (item, identifier, after) in enumerate(planned):
                words, length, targets = (int(item[key]) for key in ('word_exposures', 'input_tokens', 'next_token_loss_positions'))
                counts['word_exposures'] += words; counts['input_tokens'] += length; counts['windows'] += 1
                cursor = dict(after)
                if targets == 0:
                    counts['skipped_zero_target_input_tokens'] += length
                    counts['skipped_zero_target_word_exposures'] += words
                    zero_target_windows_since_update += 1
                    continue
                inputs = torch.tensor(np.asarray(item['input_ids'], dtype=np.int64).copy(), dtype=torch.long, device=device).unsqueeze(0)
                pending[ordinal]['forward_started'] = True; counts['forward_attempts'] += 1
                counts['forward_input_tokens'] += length; counts['forward_word_exposures'] += words
                prediction = model(inputs, aux_weight=p['aux_weight'] if mode == 'sparse' else 0.0)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                pending[ordinal]['forward_completed'] = True; counts['forward_calls'] += 1
                if prediction.token_loss_count != targets:
                    raise ValueError('Model loss-token count disagrees with fixed window accounting')
                if not bool(torch.isfinite(prediction.loss)):
                    raise FloatingPointError('Nonfinite baseline loss')
                counts['loss_tokens'] += targets
                ce_sum += float(prediction.lm_loss.detach()) * targets
                auxiliary_sum += float(prediction.aux_loss.detach()) * targets
                scaled = prediction.loss * (targets / targets_total)
                pending[ordinal]['backward_started'] = True; counts['backward_attempts'] += 1
                scaled.backward()
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                pending[ordinal]['backward_completed'] = True; counts['backward_calls'] += 1
                for stats in prediction.attention_stats:
                    for key in ('logical_kept_pairs', 'logical_dense_causal_pairs', 'allocated_main_score_elements', 'indexer_score_elements'):
                        attention_totals[key] = attention_totals.get(key, 0) + int(stats.get(key, 0))
            identifiers = [identifier for _, identifier, _ in planned]
            if targets_total:
                multiplier = word_lr_multiplier(counts['word_exposures'], p)
                for group in optimizer.param_groups:
                    base = p['indexer_learning_rate'] if group['name'] == 'indexer' else p['learning_rate']
                    group['lr'] = base * multiplier
                last_lr = {group['name']: float(group['lr']) for group in optimizer.param_groups}
                # The gradients already accumulated above are read exactly once
                # per group. These are combined-objective gradients, not an
                # attribution separating LM and auxiliary effects. No extra
                # forward/backward or in-place gradient operation is added.
                backbone_grad_norm = _gradient_l2_norm(main_params)
                indexer_grad_norm = _gradient_l2_norm(index_params)
                clip_scope = p.get('gradient_clip_scope', 'global')
                grad_norm, clip_coefficient, backbone_clip_coefficient, indexer_clip_coefficient = _clip_parameter_groups(
                    model, main_params, index_params, p['grad_clip_norm'], clip_scope)
                optimizer.step()
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                counts['updates'] += 1
                counts['scientific_updates' if p['scope'] == 'scientific' else 'engineering_updates'] += 1
                zero_target_windows_since_update = 0
                append('update', window_ids=identifiers, windows=copy.deepcopy(pending),
                       loss_token_weighted_ce=ce_sum / targets_total, loss_token_weighted_aux=auxiliary_sum / targets_total,
                       loss_tokens_this_update=targets_total, lr=dict(last_lr), lr_word_position=counts['word_exposures'],
                       lr_multiplier=multiplier, grad_norm=float(grad_norm.detach()), counts=dict(counts), cursor=dict(cursor),
                        backbone_grad_norm=backbone_grad_norm, indexer_grad_norm=indexer_grad_norm,
                        grad_clip_coefficient=clip_coefficient, grad_clip_max_norm=p['grad_clip_norm'],
                        gradient_clip_scope=clip_scope,
                        backbone_grad_clip_coefficient=backbone_clip_coefficient,
                        indexer_grad_clip_coefficient=indexer_clip_coefficient,
                        grad_clip_coefficient_semantics=('global_coefficient' if clip_scope == 'global' else 'backbone_group_coefficient'),
                        gradient_scope=('combined_objective_before_global_clipping' if clip_scope == 'global' else
                                        'combined_objective_before_separate_group_clipping'),
                       gradient_attribution='These read-only group norms do not separate LM and auxiliary gradient contributions.',
                       aux_weight=p['aux_weight'], aux_weight_applied=p['aux_weight'] if mode == 'sparse' else 0.0,
                       weighted_aux_loss=(p['aux_weight'] if mode == 'sparse' else 0.0) * auxiliary_sum / targets_total,
                       loss_token_weighted_combined_objective=(ce_sum + (p['aux_weight'] if mode == 'sparse' else 0.0) * auxiliary_sum) / targets_total,
                       attention_stats=attention_totals, update_seconds=time.perf_counter() - update_start,
                       elapsed_wall_seconds=elapsed(), estimated_gpu_usd=estimated_paid_cost())
            else:
                append('consumed_without_update', window_ids=identifiers, windows=copy.deepcopy(pending),
                       counts=dict(counts), cursor=dict(cursor), reason='zero_loss_targets')
            pending = []
            if record_milestones():
                checkpoint(reason='milestone_model_snapshot_committed')
            if targets_total and p.get('eval_manifest') and p['eval_every_updates'] and counts['updates'] % p['eval_every_updates'] == 0:
                evaluate('cadence')
            if counts['updates'] % p['checkpoint_every_updates'] == 0:
                checkpoint(reason='committed_boundary')
            if 'max_epochs' in p and cursor['epoch'] >= p['max_epochs']:
                stop_reason = 'epoch_complete'
            elif cap_pending or counts['word_exposures'] >= p['max_word_exposures']:
                stop_reason = 'word_budget_reached'
            elif zero_target_windows_since_update >= len(dataset):
                stop_reason = 'no_trainable_targets_in_complete_epoch'
        complete = (stop_reason == 'epoch_complete' if 'max_epochs' in p else
                    stop_reason in ('word_budget_reached', 'max_updates_reached'))
        if p.get('eval_final', False) and complete:
            evaluate('final')
        record_final_snapshot(stop_reason)
        append('run_stop', status=stop_reason, counts=dict(counts), eval_counts=dict(eval_counts), cursor=dict(cursor),
               elapsed_wall_seconds=elapsed(), estimated_gpu_usd=estimated_paid_cost())
        checkpoint(reason=stop_reason)
        summary = {'status': stop_reason, 'scope': p['scope'], 'mode': mode, 'protocol_sha256': phash,
                   'source_hashes': shashes, 'data_fingerprint': dataset.fingerprint,
                   'counts': dict(counts), 'eval_counts': dict(eval_counts), 'cursor': dict(cursor),
                   'checkpoint': str(checkpoint_path), 'checkpoint_sha256': _file_hash(checkpoint_path),
                   'parameter_counts': sizes, 'optimizer_groups': [group['name'] for group in optimizer.param_groups],
                   'last_lr': dict(last_lr), 'elapsed_wall_seconds': elapsed(),
                   'last_eval_point': last_eval_point, 'initial_parameter_hashes': initial_parameter_hashes,
                   'snapshot_state': copy.deepcopy(snapshot_state),
                   'checkpoint_retention': 'latest optimizer checkpoint; immutable milestone/final model snapshots; unique failure snapshots',
                   'input_accounting': 'input_tokens/word_exposures conservatively include consumed zero-target windows; forward_* counts only submitted forward inputs; skipped_zero_target_* explicitly separated',
                   'estimated_gpu_usd': estimated_paid_cost(), 'stage_spent_usd_before_run': p.get('stage_spent_usd', 0),
                   'time_limit_kind': 'cooperative_boundary_not_hardkill',
                   'cost_mode': p.get('cost_mode', 'rate_bounded' if p['device'] == 'cuda' else 'cpu_engineering'),
                   'cost_limitations': ('GPU hourly price is unknown; estimated_gpu_usd is null and no dollar ceiling is claimed. Wall-time is bounded; external billing needs reconciliation.'
                                        if p.get('cost_mode') == 'time_bounded_unknown_rate' else
                                        'Hourly estimate, not an invoice; external idle time/storage and abrupt process death need launcher reconciliation.'),
                   'quality_claim': 'No conclusion from engine completion alone', 'completed_utc': _utc()}
        _atomic_json(run_dir / 'summary.json', summary)
        return summary
    except BaseException as error:
        failures = run_dir / 'failures'
        failures.mkdir(exist_ok=True)
        identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8]
        failure = {'utc': _utc(), 'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc(), 'protocol_sha256': phash, 'source_hashes': shashes,
                   'mode': mode, 'counts': dict(counts), 'cursor': dict(cursor), 'pending_batch': copy.deepcopy(pending),
                   'active_evaluation': copy.deepcopy(active_evaluation),
                   'elapsed_wall_seconds': elapsed(), 'resume_supported': False,
                   'reason': 'Failure snapshot is evidence, not a committed replay point'}
        _atomic_json(failures / f'{identifier}.json', failure)
        if state_ready:
            append('failure', error_type=type(error).__name__, error=str(error), failure_file=str(failures / f'{identifier}.json'),
                   counts=dict(counts), cursor=dict(cursor), pending_batch=copy.deepcopy(pending))
            try:
                checkpoint(failures / f'{identifier}.pt', resumable=False, reason='exception')
            except BaseException:
                (failures / f'{identifier}-checkpoint-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        lock.unlink(missing_ok=True)
