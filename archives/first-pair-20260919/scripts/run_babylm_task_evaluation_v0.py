"""Default-plan-only, local CPU fixture runner for the BabyLM scoring adapter.

This is not a scientific evaluation launcher. CUDA and non-synthetic execution
are deliberately unsupported. No downloads, test discovery, resume or training.
The wall limit is cooperative; it cannot interrupt a single running forward.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
UPSTREAM_COMMIT = '6f825c291e2c4c78ad33b1935fd64d45f52642dc'
TOKENIZER_SHA256 = '230b9d6993dcaf32f40cec2c44ac79d7d713213616ba2d8e45ad4a96e5a9dbe6'
COUNT_KEYS = ('model_forward_attempts', 'model_forward_calls', 'backward_calls',
              'optimizer_updates', 'records', 'candidate_sequences', 'source_tokens',
              'forward_input_tokens', 'padded_forward_positions', 'scored_target_tokens',
              'submitted_forward_input_tokens', 'submitted_padded_forward_positions')


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected an externally supplied lowercase SHA256')
    return value


def local_path(value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError('Input must be an existing local file within this project')
    return path


def verified_file(value, expected):
    path = local_path(value)
    if sha(path) != require_sha(expected):
        raise ValueError(f'SHA256 mismatch: {path}')
    return path


def write_new_json(path, value):
    # Output lives in a fresh exclusively locked run directory. Exclusive file
    # creation protects earlier evidence even if a caller supplies an old name.
    with Path(path).open('xb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def append_json(path, value):
    with Path(path).open('ab') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_protocol(p):
    required = {'schema_version', 'scope', 'device', 'snapshot', 'snapshot_sha256',
                'snapshot_metadata_sha256', 'training_protocol_sha256', 'mode', 'tokenizer',
                'tokenizer_sha256', 'task_manifest', 'task_manifest_sha256', 'seed',
                'temperature', 'batch_size', 'max_items', 'max_forward_attempts',
                'max_wall_seconds', 'max_input_tokens', 'max_model_parameters'}
    if required - set(p):
        raise ValueError(f'Missing evaluation protocol fields: {sorted(required-set(p))}')
    if p['schema_version'] != 1 or p['mode'] not in ('dense', 'sparse'):
        raise ValueError('Unsupported protocol version or model mode')
    if p['device'] not in ('cpu', 'cuda'):
        raise ValueError('Device must be explicit cpu or cuda (CUDA remains plan-only)')
    for key, maximum in (('batch_size', 8), ('max_items', 128), ('max_forward_attempts', 64),
                         ('max_input_tokens', 2048), ('max_model_parameters', 1_000_000)):
        if type(p[key]) is not int or not 0 < p[key] <= maximum:
            raise ValueError(f'{key} must be an integer in [1, {maximum}]')
    if type(p['seed']) is not int or not 0 <= p['seed'] < 2**32:
        raise ValueError('The evaluation seed must be a frozen uint32 integer')
    if type(p['temperature']) not in (int, float) or p['temperature'] != 1.0:
        raise ValueError('This minimal runner supports only predeclared temperature 1')
    if type(p['max_wall_seconds']) not in (int, float) or not math.isfinite(p['max_wall_seconds']) or not 0 < p['max_wall_seconds'] <= 300:
        raise ValueError('CPU fixture wall limit must be finite and in (0, 300] seconds')
    for key in ('snapshot_sha256', 'snapshot_metadata_sha256', 'training_protocol_sha256',
                'task_manifest_sha256', 'tokenizer_sha256'):
        require_sha(p[key])
    if p['tokenizer_sha256'] != TOKENIZER_SHA256:
        raise ValueError('Only the frozen train-only tokenizer is supported')


def load_snapshot_model(p):
    """Safe model-only load: verify external hashes before restricted unpickling."""
    import torch
    from babylm_hybrid.config import HybridConfig
    from babylm_hybrid.model import build_model, parameter_counts
    from babylm_hybrid.training import source_hashes, _validate_protocol

    path = verified_file(p['snapshot'], p['snapshot_sha256'])
    metadata_path = verified_file(str(path.with_suffix('.json')), p['snapshot_metadata_sha256'])
    verified_file(p['tokenizer'], p['tokenizer_sha256'])
    metadata = strict_json(metadata_path.read_bytes().decode('utf-8'))
    if metadata.get('sha256') != p['snapshot_sha256']:
        raise ValueError('Snapshot sidecar does not identify the externally pinned weights')
    # weights_only is intentional: never enable unrestricted pickle or an
    # allowlist for a foreign checkpoint. Native model-only snapshots need none.
    saved = torch.load(path, map_location='cpu', weights_only=True)
    required = {'snapshot_schema_version', 'kind', 'resume_supported', 'point', 'counts', 'cursor',
                'mode', 'nominal_crossed_milestones', 'protocol_sha256', 'source_hashes',
                'data_fingerprint', 'initial_parameter_hashes', 'saved_utc', 'word_accounting',
                'purpose', 'protocol', 'model_state'}
    if not isinstance(saved, dict) or set(saved) != required:
        raise ValueError('Only the exact project model-only snapshot schema is accepted')
    if saved['snapshot_schema_version'] != 1 or saved['kind'] != 'model_only' or saved['resume_supported'] is not False:
        raise ValueError('Not a native immutable model-only snapshot')
    provenance = {key: value for key, value in saved.items() if key not in ('model_state', 'protocol')}
    if {key: metadata.get(key) for key in provenance} != provenance:
        raise ValueError('Snapshot sidecar and embedded provenance disagree')
    if saved['mode'] != p['mode'] or saved['protocol_sha256'] != p['training_protocol_sha256']:
        raise ValueError('Pinned model mode or training protocol hash mismatch')
    training_protocol = saved['protocol']
    if hashlib.sha256(canonical(training_protocol)).hexdigest() != saved['protocol_sha256']:
        raise ValueError('Embedded training protocol contents fail their hash')
    if training_protocol.get('scope') != 'engineering_smoke' or training_protocol.get('device') != 'cpu' or training_protocol.get('dtype') != 'float32':
        raise ValueError('This runner executes only native CPU engineering-smoke snapshots')
    _validate_protocol(training_protocol)
    if saved['source_hashes'] != source_hashes():
        raise ValueError('Snapshot source hashes differ; portability requires a separate audit')
    fingerprint = saved['data_fingerprint']
    if fingerprint.get('tokenizer_sha256') != TOKENIZER_SHA256 or fingerprint.get('manifest_sha256') != training_protocol.get('train_manifest_sha256'):
        raise ValueError('Training data/tokenizer provenance is incomplete or inconsistent')
    if set(saved['point']) != {'updates', 'word_exposures', 'input_tokens', 'loss_tokens'}:
        raise ValueError('Snapshot point is incomplete')
    if any(type(value) is not int or value < 0 or value != saved['counts'].get(key) for key, value in saved['point'].items()):
        raise ValueError('Snapshot actual counters are inconsistent')
    state = saved['model_state']
    if not isinstance(state, dict) or not state or any(not torch.is_tensor(value) for value in state.values()):
        raise ValueError('Snapshot model state must contain only named tensors')
    if sum(value.numel() for value in state.values()) > p['max_model_parameters']:
        raise ValueError('Snapshot exceeds the CPU fixture parameter/state-element bound')
    config = dict(training_protocol['model_config'])
    if set(config) != set(HybridConfig().to_dict()) or config.get('vocab_size') != 16384:
        raise ValueError('Complete native model configuration with the frozen 16384-token vocabulary required')
    # Only the one prepared engineering shape is executable. A small state dict
    # must not smuggle an enormous head count into constructor allocations.
    if canonical(config) != canonical(HybridConfig(vocab_size=16384).to_dict()):
        raise ValueError('Only the fixed HybridConfig(vocab_size=16384) CPU fixture is executable')
    config['layer_types'] = tuple(config['layer_types'])
    model = build_model(HybridConfig(**config), p['mode'], training_protocol['backbone_seed'], training_protocol['indexer_seed'])
    actual_counts = parameter_counts(model)
    if actual_counts['total'] > p['max_model_parameters']:
        raise ValueError('Constructed model exceeds the CPU fixture parameter bound')
    expected_state = model.state_dict()
    if set(state) != set(expected_state):
        raise ValueError('Snapshot state keys do not exactly match the native model')
    for key, value in state.items():
        expected = expected_state[key]
        if value.shape != expected.shape or value.dtype != expected.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Snapshot tensor shape/dtype/finiteness mismatch: {key}')
    initial_hashes = {name: hashlib.sha256(canonical({'shape': list(parameter.shape), 'dtype': str(parameter.dtype)})
                     + parameter.detach().contiguous().numpy().tobytes()).hexdigest()
                     for name, parameter in model.named_parameters()}
    if saved['initial_parameter_hashes'] != initial_hashes:
        raise ValueError('Random-initialization seed provenance mismatch')
    model.load_state_dict(state, strict=True)
    return model, {'snapshot_sha256': p['snapshot_sha256'], 'metadata_sha256': p['snapshot_metadata_sha256'],
                   'training_protocol_sha256': saved['protocol_sha256'], 'training_source_hashes': saved['source_hashes'],
                   'point': saved['point'], 'mode': saved['mode'], 'parameter_counts': actual_counts,
                   'tokenizer_sha256': TOKENIZER_SHA256}


def load_records(p):
    from babylm_hybrid.official_task_records import normalize_record
    manifest_path = verified_file(p['task_manifest'], p['task_manifest_sha256'])
    manifest = strict_json(manifest_path.read_bytes().decode('utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('scope') != 'engineering_fixture' or manifest.get('upstream_commit') != UPSTREAM_COMMIT:
        raise ValueError('Only a pinned synthetic engineering task manifest is executable')
    if not isinstance(manifest.get('files'), list) or not manifest['files']:
        raise ValueError('Manifest must explicitly list local task files')
    records, receipts, seen_paths, task = [], [], set(), None
    total_bytes = 0
    for entry in manifest['files']:
        if entry.get('synthetic') is not True or entry.get('split') != 'synthetic' or entry.get('task') not in ('blimp', 'ewok'):
            raise ValueError('Only explicitly synthetic BLiMP/EWoK fixture files are executable')
        if type(entry.get('records')) is not int or entry['records'] <= 0:
            raise ValueError('Expected source record counts must be positive integers')
        if len(records) + entry['records'] > p['max_items']:
            raise ValueError('Frozen manifest exceeds the item limit; no truncation/subset selection')
        if task is not None and task != entry['task']:
            raise ValueError('One manifest must contain exactly one task')
        task = entry['task']
        relative = Path(entry['path'])
        path = (manifest_path.parent / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(manifest_path.parent) or path in seen_paths:
            raise ValueError('Task source path escapes its manifest directory or repeats a source')
        total_bytes += path.stat().st_size
        if total_bytes > 8 * 1024 * 1024:
            raise ValueError('Synthetic source files exceed the 8 MiB bound')
        verified_file(str(path), entry['sha256'])
        seen_paths.add(path)
        read_count = 0
        # Preserve CRLF and exact source-line bytes. Unicode line separators
        # inside a JSON string are not physical JSONL record boundaries.
        source_text = path.read_bytes().decode('utf-8')
        ended_with_newline = source_text.endswith('\n')
        parts = source_text.split('\n')
        if ended_with_newline:
            parts.pop()
        for ordinal, part in enumerate(parts):
            raw_line = part + ('\n' if ordinal < len(parts)-1 or ended_with_newline else '')
            if not raw_line.strip():
                raise ValueError('Blank task line; refusing silent skipping')
            raw = strict_json(raw_line)
            if not isinstance(raw, dict) or raw.get('synthetic') is not True:
                raise ValueError('Each raw fixture record must explicitly set synthetic=true')
            read_count += 1
            if read_count > entry['records'] or len(records) >= p['max_items']:
                raise ValueError('Actual source count exceeds the frozen manifest/item cap')
            records.append(normalize_record(raw, task=task, source_path=str(path), line_number=ordinal+1,
                                            source_sha256=entry['sha256'], raw_line=raw_line,
                                            full_sentence_scores=False))
        if read_count != entry['records']:
            raise ValueError('Actual source count differs from frozen manifest')
        receipts.append({**entry, 'resolved_path': str(path), 'actual_records': read_count})
    return task, records, {'manifest_sha256': p['task_manifest_sha256'], 'manifest': manifest, 'sources': receipts}


def batches_in_order(records, batch_size):
    start = 0
    while start < len(records):
        candidates = len(records[start]['sentences'])
        end = start + 1
        while end < min(start + batch_size, len(records)) and len(records[end]['sentences']) == candidates:
            end += 1
        yield records[start:end]
        start = end


def run_evaluation(protocol_path, expected_protocol_sha256, output_dir, execute=False):
    protocol_path = verified_file(protocol_path, expected_protocol_sha256)
    p = strict_json(protocol_path.read_bytes().decode('utf-8'))
    validate_protocol(p)
    if not execute:
        return {'status': 'plan_only_no_model_or_task_data_loaded', 'execute_supported': p['scope'] == 'engineering_fixture' and p['device'] == 'cpu',
                'scope': p['scope'], 'device': p['device'], 'protocol_sha256': expected_protocol_sha256,
                'limits': {key: p[key] for key in ('batch_size', 'max_items', 'max_forward_attempts', 'max_wall_seconds', 'max_model_parameters')},
                'scientific_ready': False, 'gpu_launch_supported': False}
    if p['scope'] != 'engineering_fixture' or p['device'] != 'cpu':
        raise ValueError('Execute supports CPU engineering_fixture only; no paid/GPU/scientific launch path')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'run.lock'
    # Acquiring before examining files prevents deleting someone else's lock.
    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'utc': utc()}, stream)
    started = time.perf_counter()
    counts = {key: 0 for key in COUNT_KEYS}
    finished_records, combined_scores = [], []
    combined_target_counts, combined_ranking, combined_scorable, completed_batch_sizes = [], [], [], []
    identity = None
    task = None
    active_batch = None
    output_ready = False

    def elapsed():
        return time.perf_counter() - started

    def check_time():
        if elapsed() >= p['max_wall_seconds']:
            raise TimeoutError('Cooperative evaluation wall limit reached; no later batch started')

    def add_counts(values):
        for key in COUNT_KEYS:
            value = values.get(key, 0)
            if type(value) is not int or value < 0:
                raise ValueError(f'Invalid scorer accounting: {key}')
            counts[key] += value

    def aggregate_completed():
        from babylm_hybrid.official_task_records import aggregate_scores
        return aggregate_scores(finished_records, {'task': task, 'upstream_commit': UPSTREAM_COMMIT,
                                'temperatures': [1.0], 'scores': [combined_scores],
                                'normalization': 'completion_token_sum',
                                'scored_target_counts': combined_target_counts,
                                'scorable_candidates': combined_scorable,
                                'ranking_allowed_per_record': combined_ranking,
                                'record_ids': [r['record_id'] for r in finished_records],
                                'batch_sizes': completed_batch_sizes}, tie_seed=p['seed'])

    try:
        if any(path != lock for path in output.iterdir()):
            raise ValueError('Output already contains evidence; use a fresh directory, no resume/overwrite')
        output_ready = True
        write_new_json(output / 'protocol.json', p)
        write_new_json(output / 'start.json', {'utc': utc(), 'protocol_sha256': expected_protocol_sha256,
                       'snapshot_sha256': p['snapshot_sha256'], 'scope': 'engineering_fixture',
                       'evaluation_source_hashes': {str(path.relative_to(ROOT)): sha(path) for path in
                           (Path(__file__), ROOT / 'src/babylm_hybrid/official_causal_scoring.py',
                            ROOT / 'src/babylm_hybrid/official_task_records.py')},
                       'time_limit_kind': 'cooperative_between_batches_not_hard_timeout'})
        task, records, receipt = load_records(p)
        frozen_batches = list(batches_in_order(records, p['batch_size']))
        planned_forwards = sum(len(batch[0]['sentences']) for batch in frozen_batches)
        if planned_forwards > p['max_forward_attempts']:
            raise ValueError('Frozen task plan exceeds the forward limit; no partial subset execution')
        write_new_json(output / 'inputs.json', receipt)
        for record in records:
            append_json(output / 'records.jsonl', record)
        check_time()
        model, identity = load_snapshot_model(p)
        from babylm_hybrid.official_causal_scoring import FixedLocalTokenizer, score_records, CausalScoringError
        import torch
        import numpy as np
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        random.seed(p['seed']); np.random.seed(p['seed']); torch.manual_seed(p['seed'])
        processor = FixedLocalTokenizer(local_path(p['tokenizer']))
        write_new_json(output / 'model-identity.json', identity)
        for batch_index, batch in enumerate(frozen_batches):
            check_time()
            needed = len(batch[0]['sentences'])
            if counts['model_forward_attempts'] + needed > p['max_forward_attempts']:
                raise RuntimeError('Forward limit reached before the next whole batch')
            active_batch = {'batch_index': batch_index, 'record_ids': [r['record_id'] for r in batch], 'started_utc': utc()}
            try:
                scores = score_records(model, processor, batch, task, device='cpu', temperatures=(1.0,), max_input_tokens=p['max_input_tokens'])
            except CausalScoringError as error:
                add_counts(error.counters)
                active_batch.update(failed_candidate_index=error.candidate_index,
                                    incomplete_candidate_scores_not_exposed_by_scorer=True)
                raise
            add_counts(scores['counters'])
            # Preserve scores before any aggregation or ranking can fail.
            append_json(output / 'raw-scores.jsonl', {**active_batch, 'completed_utc': utc(),
                        'snapshot_sha256': p['snapshot_sha256'], 'result': scores})
            if (scores.get('task') != task or scores.get('upstream_commit') != UPSTREAM_COMMIT or
                    scores.get('normalization') != 'completion_token_sum' or scores.get('tokenizer_sha256') != TOKENIZER_SHA256 or
                    scores['temperatures'] != [1.0] or len(scores['scores']) != 1 or len(scores['scores'][0]) != len(batch)):
                raise ValueError('Scorer result order or temperature contract mismatch')
            if scores['counters']['model_forward_attempts'] != needed or scores['counters']['model_forward_calls'] != needed:
                raise ValueError('Scorer forward accounting differs from the fixed candidate plan')
            finished_records.extend(batch)
            combined_scores.extend(scores['scores'][0])
            combined_target_counts.extend(scores['scored_target_counts'])
            combined_ranking.extend(scores['ranking_allowed_per_record'])
            combined_scorable.extend(scores['scorable_candidates'])
            completed_batch_sizes.append(len(batch))
            append_json(output / 'events.jsonl', {'utc': utc(), 'event': 'batch_completed',
                        'batch_index': batch_index, 'counts': dict(counts), 'elapsed_wall_seconds': elapsed()})
            active_batch = None
        check_time()
        aggregation = aggregate_completed()
        write_new_json(output / 'predictions-and-aggregation.json', aggregation)
        summary = {'status': 'cpu_fixture_scoring_complete', 'scope': 'engineering_fixture', 'completed_utc': utc(),
                   'protocol_sha256': expected_protocol_sha256, 'snapshot_sha256': p['snapshot_sha256'],
                   'identity': identity, 'task': task, 'records_completed': len(finished_records), 'counts': counts,
                   'elapsed_wall_seconds': elapsed(), 'official_benchmark_executed': False, 'scientific_ready': False,
                   'time_limit_kind': 'cooperative_between_batches_not_hard_timeout'}
        write_new_json(output / 'summary.json', summary)
        return summary
    except BaseException as error:
        if output_ready:
            failure = {'status': 'failed_no_items_skipped', 'utc': utc(), 'error_type': type(error).__name__,
                       'error': str(error), 'traceback': traceback.format_exc(), 'counts': dict(counts),
                       'active_batch': active_batch, 'records_completed': len(finished_records),
                       'protocol_sha256': expected_protocol_sha256, 'snapshot_sha256': p['snapshot_sha256'],
                       'elapsed_wall_seconds': elapsed(), 'resume_supported': False, 'official_benchmark_executed': False}
            write_new_json(output / 'failure.json', failure)
            if finished_records:
                try:
                    write_new_json(output / 'partial-predictions-and-aggregation.json', aggregate_completed())
                except BaseException as aggregation_error:
                    write_new_json(output / 'partial-aggregation-failure.json', {'error_type': type(aggregation_error).__name__,
                                   'error': str(aggregation_error), 'traceback': traceback.format_exc()})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_evaluation(args.protocol, args.protocol_sha256, args.output_dir, args.execute), indent=2))


if __name__ == '__main__':
    main()
