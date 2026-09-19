"""Bounded, fresh E-only launcher on a second Pod; default is plan-only.

Preflight is external: require original-threshold tiny checks and a complete
12-update E engineering run on this Pod, replayed against archived first-Pod
evidence. This does not invent a same-Pod D/E preflight or reuse its weights.
Receipt SHA256 fixes an audit statement; it is not a cryptographic signature.
No downloads, renting, automatic resume, or automatic Pod shutdown.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_babylm_scientific_pair_v0 as pair

canonical, sha, load, write, utc, protocol_hash = (
    pair.canonical, pair.sha, pair.load, pair.write, pair.utc, pair.protocol_hash)
SCIENCE_FIELDS = pair.MATCH_FIELDS + (
    'schema_version', 'scope', 'max_updates', 'checkpoint_every_updates',
    'eval_manifest', 'eval_manifest_sha256', 'eval_window_indices',
    'eval_every_updates', 'eval_initial', 'eval_final',
    'milestone_word_exposures', 'eval_position_diagnostics', 'expected_stop_reason')
THRESHOLDS = {'cpu_cuda_fp32': {'atol': 1e-4, 'rtol': 1e-3},
              'gpu_full_support': {'atol': 1e-6, 'rtol': 1e-5},
              'checkpoint_replay': {'atol': 1e-6, 'rtol': 1e-5}}
REPLAY_FIELDS = ('loss_token_weighted_ce', 'loss_token_weighted_aux',
                 'loss_token_weighted_combined_objective', 'grad_norm',
                 'backbone_grad_norm', 'indexer_grad_norm', 'grad_clip_coefficient')
NUMERICAL_CHECKS = ('shared_seeded_backbone_exact', 'cpu_cuda_fp32_dense',
                    'cpu_cuda_fp32_sparse', 'truly_sparse_gpu_support',
                    'sparse_gpu_gradient_groups',
                    'gpu_full_support_dense_sparse_equivalence',
                    'checkpoint_optimizer_rng_replay')


def require(value, message):
    if not value:
        raise ValueError(message)


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def checked_file(path, expected):
    path = resolve(path)
    require(isinstance(expected, str) and len(expected) == 64 and sha(path) == expected,
            f'Pinned SHA256 mismatch: {path}')
    return path


def evidence(p, prefix):
    path = checked_file(p[prefix + '_path'], p[prefix + '_sha256'])
    return load(path)


def portable_name(value):
    value = str(value).replace('\\', '/')
    for marker in ('src/babylm_hybrid/', 'scripts/', 'data/', 'transformers/models/'):
        if value.startswith(marker):
            return value
        if '/' + marker in value:
            return marker + value.rsplit('/' + marker, 1)[1]
    raise ValueError(f'Unrecognized evidence source/data path: {value}')


def portable_hashes(mapping):
    require(isinstance(mapping, dict) and bool(mapping), 'Nonempty hash mapping required')
    result = {}
    for path, digest in mapping.items():
        name = portable_name(path)
        require(name not in result and isinstance(digest, str) and len(digest) == 64,
                'Duplicate portable path or invalid SHA256')
        result[name] = digest
    return result


def verify_current_sources(mapping):
    portable_hashes(mapping)
    for path, digest in mapping.items():
        checked_file(path, digest)


def portable_data(value):
    require(isinstance(value, dict), 'Data fingerprint required')
    return {**value, 'manifest_path': portable_name(value['manifest_path']),
            'artifacts': portable_hashes(value['artifacts'])}


def compare_fields(a, b, fields):
    for key in fields:
        require(key in a and key in b and canonical(a[key]) == canonical(b[key]),
                f'Frozen recipe field mismatch: {key}')


def verify_release(p, parent):
    receipt = evidence(p, 'old_sparse_release')
    require(receipt.get('schema_version') == 1 and receipt.get('kind') == 'old_sparse_schedule_release',
            'Old E scheduling release schema missing')
    require(receipt.get('parent_scientific_protocol_sha256') == p['parent_scientific_protocol_sha256']
            and receipt.get('old_pod_id') == parent.get('pod_id')
            and receipt.get('new_pod_id') == p.get('pod_id')
            and receipt.get('old_sparse_schedule_disabled') is True
            and receipt.get('old_sparse_not_running') is True
            and isinstance(receipt.get('verified_utc'), str) and bool(receipt['verified_utc']),
            'Old E scheduling is not explicitly released for this parent/new Pod')
    return receipt


def verify_numerics(p):
    report = evidence(p, 'numerics')
    require(report.get('scope') == 'tiny_synthetic_gpu_numerics_engineering_only'
            and report.get('status') == 'passed' and report.get('passed') is True,
            'Tiny numerical preflight did not pass')
    require(report.get('thresholds') == THRESHOLDS, 'Original numerical thresholds changed')
    checks = report.get('checks', {})
    require(set(checks) == set(NUMERICAL_CHECKS), 'Incomplete numerical check suite')
    def visit(value, expected_threshold=None):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('passed', 'finite'):
                    require(item is True, 'Nested numerical check failed')
                if key == 'threshold':
                    require(item == expected_threshold, 'Nested numerical threshold changed')
                if key in ('failed_names', 'missing_gradients', 'nonfinite_gradients', 'zero_gradient_names'):
                    require(item == [], 'Numerical failure details are nonempty')
                visit(item, expected_threshold)
        elif isinstance(value, list):
            for item in value:
                visit(item, expected_threshold)
        elif isinstance(value, float):
            require(math.isfinite(value), 'Nonfinite numerical evidence')
    for key in NUMERICAL_CHECKS:
        require(checks[key].get('passed') is True, f'Missing pass: {key}')
        threshold = (THRESHOLDS['cpu_cuda_fp32'] if key.startswith('cpu_cuda_fp32_') else
                     THRESHOLDS['gpu_full_support'] if key.startswith('gpu_full_support_') else
                     THRESHOLDS['checkpoint_replay'] if key.startswith('checkpoint_') else None)
        visit(checks[key], threshold)
    c = report['counts']
    for key, expected in {'model_forward_attempts': 9, 'model_forward_calls': 9,
                          'backward_attempts': 9, 'backward_calls': 9,
                          'cpu_model_forward_calls': 2, 'cuda_model_forward_calls': 7,
                          'cpu_backward_calls': 2, 'cuda_backward_calls': 7,
                          'optimizer_step_attempts': 3, 'engineering_optimizer_steps': 3,
                          'scientific_optimizer_steps': 0, 'scientific_training_tokens': 0}.items():
        require(c.get(key) == expected, f'Unexpected numerical accounting: {key}')
    verify_current_sources(report['source_hashes'])
    require(os.environ.get('RUNPOD_POD_ID') == p.get('pod_id') and bool(p.get('pod_id')),
            'Current RUNPOD_POD_ID differs from the new protocol')
    gpu = report['hardware']['nvidia_smi']
    rows = list(csv.reader(io.StringIO(gpu.get('stdout', ''))))
    require(gpu.get('returncode') == 0 and len(rows) == 1 and len(rows[0]) >= 4,
            'Require one GPU UUID/driver numerical receipt')
    expected = (rows[0][2].strip(), rows[0][3].strip())
    current = subprocess.run(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=10, check=False)
    observed = list(csv.reader(io.StringIO(current.stdout)))
    require(current.returncode == 0 and len(observed) == 1
            and tuple(cell.strip() for cell in observed[0]) == expected,
            'Current GPU UUID/driver differs from the new numerical preflight')
    return report


def read_preflight(p, prefix, mode, with_events=False):
    q = evidence(p, prefix + '_protocol')
    s = evidence(p, prefix + '_summary')
    require(q.get('scope') == s.get('scope') == 'gpu_preflight'
            and s.get('mode') == mode and s.get('status') == 'max_updates_reached'
            and q.get('max_updates') == 12 and s.get('counts', {}).get('updates') == 12
            and s['counts'].get('engineering_updates') == 12
            and s['counts'].get('scientific_updates') == 0
            and s.get('protocol_sha256') == protocol_hash(q), 'Invalid 12-update engineering summary')
    compare_fields(q, p, pair.MATCH_FIELDS)
    require(not q.get('eval_initial') and not q.get('eval_final') and q.get('eval_every_updates') == 0,
            'Engineering replay must have no evaluation work')
    require(all(v == 0 for v in s['eval_counts'].values()), 'Unexpected engineering evaluation counts')
    require(s.get('initial_parameter_hashes') and s.get('parameter_counts'), 'Missing initialization/size evidence')
    if not with_events:
        return q, s, None
    path = checked_file(p[prefix + '_events_path'], p[prefix + '_events_sha256'])
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    require(rows and all(row.get('event_id') == i + 1 for i, row in enumerate(rows)), 'Event sequence mismatch')
    require(rows[0].get('type') == 'run_start' and rows[-1].get('type') == 'run_stop'
            and all(row.get('mode') == mode for row in rows)
            and not any(row.get('type') in ('failure', 'resume') for row in rows), 'Invalid fresh engineering event history')
    start, stop = rows[0], rows[-1]
    for key in ('protocol_sha256', 'source_hashes', 'data_fingerprint', 'initial_parameter_hashes'):
        require(start.get(key) == s.get(key), f'Engineering start/summary mismatch: {key}')
    require(stop.get('counts') == s['counts'] and stop.get('status') == s['status'], 'Engineering stop/summary mismatch')
    updates = [row for row in rows if row['type'] == 'update']
    require(len(updates) == 12 and [row['counts']['updates'] for row in updates] == list(range(1, 13)),
            'Exactly 12 unique engineering updates required')
    return q, s, updates


def verify_preflight(p, parent, audit):
    numerical = verify_numerics(p)
    _, dense, _ = read_preflight(p, 'reference_dense_preflight', 'dense')
    _, sparse, reference = read_preflight(p, 'reference_sparse_preflight', 'sparse', True)
    q, new, observed = read_preflight(p, 'new_sparse_preflight', 'sparse', True)
    require(q.get('pod_id') == p['pod_id'] and p['pod_id'] != parent['pod_id'], 'New E must identify a different Pod')
    require(pair.shared_hashes(dense) == pair.shared_hashes(sparse) == pair.shared_hashes(new),
            'D/E random backbone differs')
    require(sparse['initial_parameter_hashes'] == new['initial_parameter_hashes']
            and any('.indexer.' in key for key in new['initial_parameter_hashes']), 'Sparse/indexer initialization differs')
    require(dense['counts'] == sparse['counts'] == new['counts']
            and dense['cursor'] == sparse['cursor'] == new['cursor'], 'Engineering data exposures/order boundary differ')
    require(sparse['parameter_counts'] == new['parameter_counts'], 'Candidate parameter counts differ')
    sources = portable_hashes(new['source_hashes'])
    require(sources == portable_hashes(dense['source_hashes']) == portable_hashes(sparse['source_hashes']),
            'First/new Pod training source contents differ')
    verify_current_sources(new['source_hashes'])
    nsources = portable_hashes(numerical['source_hashes'])
    require(all(sources[name] == value for name, value in nsources.items() if name in sources),
            'Tiny checker/model source differs from full preflight')
    data = portable_data(new['data_fingerprint'])
    require(data == portable_data(dense['data_fingerprint']) == portable_data(sparse['data_fingerprint'])
            and data['manifest_sha256'] == p['train_manifest_sha256'], 'First/new Pod data contents differ')
    checked_file(p['train_manifest'], p['train_manifest_sha256'])
    checked_file(p['eval_manifest'], p['eval_manifest_sha256'])
    for path, digest in new['data_fingerprint']['artifacts'].items():
        checked_file(path, digest)
    audit.update(threshold=THRESHOLDS['cpu_cuda_fp32'], comparisons=[],
                 interpretation='Scalar trajectory replay, not elementwise gradient equivalence or speed transfer')
    for expected, actual in zip(reference, observed):
        for key in ('window_ids', 'counts', 'cursor', 'lr', 'lr_word_position', 'loss_tokens_this_update'):
            require(expected.get(key) == actual.get(key), f'Engineering replay exact field mismatch: {key}')
        for key in REPLAY_FIELDS:
            x, y = expected.get(key), actual.get(key)
            require(type(x) in (int, float) and type(y) in (int, float) and math.isfinite(x) and math.isfinite(y),
                    f'Nonfinite/missing replay scalar: {key}')
            tolerance = 1e-4 + 1e-3 * abs(x)
            audit['comparisons'].append({'update': expected['counts']['updates'], 'field': key,
                'reference': x, 'observed': y, 'absolute_error': abs(y-x), 'tolerance': tolerance,
                'passed': abs(y-x) <= tolerance})
    audit['passed'] = all(row['passed'] for row in audit['comparisons'])
    require(audit['passed'], 'Original-threshold full-candidate replay failed; do not relax or launch')
    return new


def launch(protocol_path, expected_sha256, output_dir, execute=False):
    protocol_path = checked_file(protocol_path, expected_sha256)
    p = load(protocol_path)
    parent = evidence(p, 'parent_scientific_protocol')
    compare_fields(p, parent, SCIENCE_FIELDS)
    require(p.get('scope') == 'scientific' and p.get('device') == 'cuda' and p.get('dtype') == 'float32',
            'Scientific CUDA FP32 required')
    require(not any(p.get(k) for k in ('resume_checkpoint', 'pretrained_model', 'initial_checkpoint')),
            'Fresh random initialization only')
    values = [float(p[k]) for k in ('hourly_rate_usd', 'stage_spent_usd', 'single_pod_ceiling_usd',
                                   'paid_ceiling_usd', 'hard_timeout_seconds_per_run', 'max_wall_seconds')]
    rate, spent, cap, engine_cap, hard, soft = values
    require(all(math.isfinite(v) for v in values) and rate > 0 and 0 <= spent < cap <= 75
            and cap == engine_cap and 0 < soft <= hard <= 259200, 'Invalid bounded 72h/USD75 Pod budget')
    if not execute:
        return {'status': 'plan_only_no_processes_started', 'protocol_sha256': expected_sha256,
                'parent_protocol_sha256': p['parent_scientific_protocol_sha256'],
                'stages': ['verify_release', 'verify_new_pod_tiny_and_sparse_replay', 'fresh_sparse'],
                'resume_supported': False, 'launch_allowed': p.get('launch_allowed', False)}
    require(p.get('launch_allowed') is True, 'Explicit launch_allowed required')
    output = resolve(output_dir)
    require(not output.exists() or not any(output.iterdir()), 'Fresh empty output required')
    locks, record = [], None
    begin = time.perf_counter()
    def cost():
        return spent + (time.perf_counter() - begin) * rate / 3600
    try:
        for name in ('babylm-scientific-pair.lock', 'babylm-gpu-preflight.lock', 'babylm-sparse-parallel.lock'):
            path = ROOT / 'logs' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as stream:
                json.dump({'pid': os.getpid(), 'utc': utc(), 'output': str(output)}, stream)
            locks.append(path)
        output.mkdir(parents=True, exist_ok=True)
        require(not any(output.iterdir()), 'Output changed while acquiring locks')
        record = {'scope': 'scientific_sparse_parallel', 'status': 'verifying', 'started_utc': utc(),
                  'protocol_sha256': expected_sha256, 'parent_protocol_sha256': p['parent_scientific_protocol_sha256'],
                  'new_pod_id': p['pod_id'], 'jobs': [], 'preflight_replay': {}, 'single_pod_ceiling_usd': cap,
                  'stage_spent_usd_before': spent, 'hourly_rate_usd': rate,
                  'cost_limitations': 'Quoted wall estimate including setup, not invoice. Pod keeps billing after exit.'}
        write(output / 'stage.json', record)
        (output / 'input-protocol.json').write_bytes(protocol_path.read_bytes())
        record['old_sparse_release'] = verify_release(p, parent)
        reference = verify_preflight(p, parent, record['preflight_replay'])
        checked_file(protocol_path, expected_sha256)
        evidence(p, 'parent_scientific_protocol')
        verify_release(p, parent)
        timeout = min(hard, (cap - cost()) / rate * 3600 - 10)
        require(timeout >= 1, 'Pod cap exhausted during evidence verification')
        child = dict(p, stage_spent_usd=cost())
        child_path = output / 'sparse-protocol.json'
        write(child_path, child)
        command = [sys.executable, str(ROOT / 'scripts/run_babylm_de_v0.py'), '--protocol', str(child_path),
                   '--mode', 'sparse', '--output-dir', str(output / 'sparse')]
        job = {'label': 'sparse', 'started_utc': utc(), 'hard_timeout_seconds': timeout,
               'child_protocol_sha256': protocol_hash(child), 'child_protocol_file_sha256': sha(child_path)}
        record['jobs'].append(job); record['status'] = 'running_sparse'
        write(output / 'stage.json', record)
        try:
            with (output / 'sparse.stdout.log').open('xb') as stdout, (output / 'sparse.stderr.log').open('xb') as stderr:
                done = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            job['returncode'] = done.returncode
            require(done.returncode == 0, 'Sparse child failed; preserve all evidence')
            summary_path = output / 'sparse/summary.json'
            summary = load(summary_path)
            require(summary.get('scope') == 'scientific' and summary.get('mode') == 'sparse'
                    and summary.get('status') == p['expected_stop_reason']
                    and summary.get('protocol_sha256') == protocol_hash(child), 'Scientific completion boundary mismatch')
            for key in ('source_hashes', 'data_fingerprint', 'initial_parameter_hashes', 'parameter_counts'):
                require(summary.get(key) == reference[key], f'Fresh sparse provenance mismatch: {key}')
            counts = summary['counts']
            require(counts.get('engineering_updates') == 0 and counts.get('scientific_updates') == counts.get('updates')
                    and type(counts.get('updates')) is int and counts['updates'] > 0
                    and 0 < counts.get('word_exposures', 0) <= p['max_word_exposures'], 'Invalid scientific counters')
            job.update(status='completed', summary_sha256=sha(summary_path), counts=counts,
                       eval_counts=summary.get('eval_counts', {}))
            record['status'] = 'scientific_sparse_complete_quality_not_yet_adjudicated'
            return record
        except subprocess.TimeoutExpired:
            job['status'] = 'hard_timeout_child_killed_and_waited'
            raise
        finally:
            job['completed_utc'] = utc()
    except BaseException as error:
        if record is not None:
            record.update(status='stopped_with_failure', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if record is not None:
                record.update(completed_utc=utc(), elapsed_seconds=time.perf_counter()-begin, estimated_stage_cost_usd=cost())
                write(output / 'stage.json', record)
        finally:
            for path in reversed(locks):
                path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(launch(args.protocol, args.protocol_sha256, args.output_dir, args.execute), indent=2))


if __name__ == '__main__':
    main()
