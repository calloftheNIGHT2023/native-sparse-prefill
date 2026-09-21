"""Explicit, sequential fresh D/E science launcher after same-Pod preflight.

Default is plan-only. No renting, automatic resume, preflight-weight loading,
or unbounded budget. Subprocess timeout kills/waits the direct Python child;
the current training entrypoint creates no worker processes. Pod billing does
not stop automatically when this launcher exits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
MATCH_FIELDS = ('model_config', 'backbone_seed', 'indexer_seed', 'data_order_seed',
                'train_manifest', 'train_manifest_sha256', 'device', 'dtype',
                'max_word_exposures', 'windows_per_update', 'learning_rate',
                'indexer_learning_rate', 'weight_decay', 'betas', 'eps',
                'grad_clip_norm', 'aux_weight', 'warmup_word_exposures', 'min_lr_ratio')


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def checked_file(path, expected):
    path = resolve(path)
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise ValueError(f'Pinned SHA256 mismatch: {path}')
    return path


def write(path, value):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def protocol_hash(p):
    return hashlib.sha256(canonical(p)).hexdigest()


def check_sources(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('Preflight source hashes are required')
    for path, expected in mapping.items():
        checked_file(path, expected)


def shared_hashes(summary):
    values = summary.get('initial_parameter_hashes')
    if not isinstance(values, dict) or not values:
        raise ValueError('Preflight random initialization provenance is missing')
    return {key: value for key, value in values.items() if '.indexer.' not in key}


def verify_preflight(p):
    stage_path = checked_file(p['preflight_stage_path'], p['preflight_stage_sha256'])
    base = stage_path.parent
    stage = load(stage_path)
    if stage.get('status') != 'paired_preflight_complete_not_scientific_training':
        raise ValueError('Same-Pod paired preflight has not completed')
    jobs = stage.get('jobs', [])
    if ([job.get('label') for job in jobs] != ['gpu-numerics', 'dense', 'sparse'] or
            any(job.get('returncode') != 0 or 'timeout' in job.get('status', '') for job in jobs)):
        raise ValueError('Preflight jobs did not all complete successfully in order')
    numerical_path = checked_file(base / 'gpu-numerics.json', p['preflight_numerics_sha256'])
    numerical = load(numerical_path)
    if numerical.get('passed') is not True or numerical.get('status') != 'passed':
        raise ValueError('Numerical preflight must explicitly pass')
    check_sources(numerical.get('source_hashes'))
    sources = None
    initial = None
    data = None
    receipts = {'stage': str(stage_path), 'stage_sha256': p['preflight_stage_sha256'],
                'numerics_sha256': p['preflight_numerics_sha256'], 'runs': {}}
    for mode in ('dense', 'sparse'):
        training_protocol_path = base / f'{mode}-protocol.json'
        q = load(training_protocol_path)
        summary_path = base / mode / 'summary.json'
        summary = load(summary_path)
        if (q.get('scope') != 'gpu_preflight' or summary.get('scope') != 'gpu_preflight' or
                summary.get('mode') != mode or summary.get('status') != 'max_updates_reached' or
                summary.get('counts', {}).get('updates') != q.get('max_updates')):
            raise ValueError(f'Incomplete or mismatched {mode} preflight summary')
        if summary.get('protocol_sha256') != protocol_hash(q):
            raise ValueError('Preflight child protocol differs from its saved summary')
        for key in MATCH_FIELDS:
            if key not in q or key not in p or canonical(q[key]) != canonical(p[key]):
                raise ValueError(f'Scientific/preflight {key} mismatch')
        check_sources(summary.get('source_hashes'))
        if sources is not None and sources != summary['source_hashes']:
            raise ValueError('D/E preflight used different source versions')
        if initial is not None and initial != shared_hashes(summary):
            raise ValueError('D/E preflight did not share the random backbone')
        if data is not None and data != summary.get('data_fingerprint'):
            raise ValueError('D/E preflight did not share fixed data')
        sources, initial, data = summary['source_hashes'], shared_hashes(summary), summary['data_fingerprint']
        receipts['runs'][mode] = {'summary_sha256': sha(summary_path), 'protocol_file_sha256': sha(training_protocol_path)}
    checked_file(p['train_manifest'], p['train_manifest_sha256'])
    if p.get('eval_manifest'):
        checked_file(p['eval_manifest'], p['eval_manifest_sha256'])
    # Data reader will rehash every stream as well; validate recorded preflight
    # artifact fingerprints here before any expensive child is started.
    if data.get('manifest_sha256') != p['train_manifest_sha256']:
        raise ValueError('Preflight data fingerprint differs from the scientific manifest')
    for path, expected in data.get('artifacts', {}).items():
        checked_file(path, expected)
    if not p.get('pod_id') or os.environ.get('RUNPOD_POD_ID') != p['pod_id']:
        raise ValueError('RUNPOD_POD_ID must identify the protocol Pod; parent may set its verified value')
    gpu_record = numerical.get('hardware', {}).get('nvidia_smi', {})
    if gpu_record.get('returncode') != 0:
        raise ValueError('Preflight lacks a successful GPU UUID/driver receipt')
    rows = list(csv.reader(io.StringIO(gpu_record.get('stdout', ''))))
    if not rows or any(len(row) < 4 for row in rows):
        raise ValueError('Malformed preflight GPU UUID/driver receipt')
    expected_gpu = sorted((row[2].strip(), row[3].strip()) for row in rows)
    current = subprocess.run(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=10, check=False)
    observed = sorted(tuple(cell.strip() for cell in row) for row in csv.reader(io.StringIO(current.stdout)))
    if current.returncode or observed != expected_gpu:
        raise ValueError('GPU UUID or driver differs from the completed preflight')
    receipts.update(pod_id=p['pod_id'], gpu_uuid_driver=expected_gpu)
    return receipts, sources, initial, data


def launch(protocol_path, expected_sha256, output_dir, execute=False):
    protocol_path = checked_file(protocol_path, expected_sha256)
    p = load(protocol_path)
    if p.get('schema_version') != 1 or p.get('scope') != 'scientific' or p.get('device') != 'cuda' or p.get('dtype') != 'float32':
        raise ValueError('Only the explicit scientific CUDA FP32 protocol is supported')
    if not execute:
        return {'status': 'plan_only_no_processes_started', 'protocol_sha256': expected_sha256,
                'stages': ['verify_same_pod_preflight', 'fresh_dense', 'fresh_sparse'],
                'launch_allowed': p.get('launch_allowed', False), 'resume_supported': False}
    if p.get('launch_allowed') is not True:
        raise ValueError('Explicit scientific launch_allowed=true is required')
    rate, spent, ceiling, engine_ceiling, hard, soft = [float(p[key]) for key in
        ('hourly_rate_usd', 'stage_spent_usd', 'pair_stage_ceiling_usd', 'paid_ceiling_usd',
         'hard_timeout_seconds_per_run', 'max_wall_seconds')]
    if (not all(math.isfinite(v) for v in (rate, spent, ceiling, engine_ceiling, hard, soft)) or
            rate <= 0 or not 0 <= spent < ceiling or ceiling != engine_ceiling or not 0 < soft <= hard):
        raise ValueError('Explicit finite rate, unexhausted whole-pair ceiling and bounded per-run time are required')
    expected_stop = p.get('expected_stop_reason')
    if expected_stop not in ('word_budget_reached', 'max_updates_reached'):
        raise ValueError('Freeze the expected scientific completion boundary')
    if any(p.get(key) for key in ('resume_checkpoint', 'pretrained_model', 'initial_checkpoint')):
        raise ValueError('Fresh random initialization only; no resume/preflight weights')
    output = resolve(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Fresh scientific pair output required; no automatic resume or overwrite')
    locks = []
    record = None
    started = time.perf_counter()
    def stage_cost():
        return spent + (time.perf_counter() - started) * rate / 3600
    try:
        # Reserve both launcher namespaces. Existing preflight launches also use
        # this lock, so they cannot begin while a scientific pair owns the GPU.
        for name in ('babylm-scientific-pair.lock', 'babylm-gpu-preflight.lock'):
            lock = ROOT / 'logs' / name
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open('x', encoding='utf-8') as stream:
                json.dump({'pid': os.getpid(), 'utc': utc(), 'output': str(output)}, stream)
            locks.append(lock)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError('Output changed while acquiring exclusive locks')
        record = {'status': 'verifying_preflight', 'started_utc': utc(), 'scope': 'scientific_pair',
                  'protocol_sha256': expected_sha256, 'jobs': [], 'whole_pair_ceiling_usd': ceiling,
                  'stage_spent_usd_before': spent, 'hourly_rate_usd': rate,
                  'historical_balance_not_a_launch_gate': True,
                  'cost_limitations': 'Quoted-rate wall estimate including supplied setup spend; not an invoice. Pod continues billing after exit.'}
        write(output / 'stage.json', record)
        (output / 'input-protocol.json').write_bytes(protocol_path.read_bytes())
        receipts, sources, initial, data = verify_preflight(p)
        record['preflight'] = receipts
        summaries = {}
        for mode in ('dense', 'sparse'):
            checked_file(protocol_path, expected_sha256)
            check_sources(sources)
            available = (ceiling - stage_cost()) / rate * 3600
            timeout = min(hard, available - 10.0)
            if timeout < 1:
                raise RuntimeError('Whole-pair ceiling reached before next child')
            child_protocol = dict(p, stage_spent_usd=stage_cost())
            child_path = output / f'{mode}-protocol.json'
            write(child_path, child_protocol)
            command = [sys.executable, str(ROOT / 'scripts/run_babylm_de_v0.py'), '--protocol', str(child_path),
                       '--mode', mode, '--output-dir', str(output / mode)]
            job = {'label': mode, 'started_utc': utc(), 'hard_timeout_seconds': timeout,
                   'child_protocol_sha256': protocol_hash(child_protocol), 'child_protocol_file_sha256': sha(child_path)}
            record['jobs'].append(job)
            record['status'] = f'running_{mode}'
            write(output / 'stage.json', record)
            begin = time.perf_counter()
            try:
                with (output / f'{mode}.stdout.log').open('xb') as stdout, (output / f'{mode}.stderr.log').open('xb') as stderr:
                    completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
                job['returncode'] = completed.returncode
                if completed.returncode:
                    raise RuntimeError(f'{mode} failed; preserve evidence and do not launch the next group')
                summary_path = output / mode / 'summary.json'
                summary = load(summary_path)
                if (summary.get('scope') != 'scientific' or summary.get('mode') != mode or
                        summary.get('status') != expected_stop or summary.get('protocol_sha256') != protocol_hash(child_protocol)):
                    raise RuntimeError(f'{mode} did not reach its frozen scientific boundary')
                if summary.get('source_hashes') != sources or summary.get('data_fingerprint') != data or shared_hashes(summary) != initial:
                    raise RuntimeError(f'{mode} source/data/random initialization differs from preflight')
                if summary.get('counts', {}).get('engineering_updates') != 0:
                    raise RuntimeError('Scientific run contains engineering/preflight updates')
                summaries[mode] = summary
                job.update(status='completed', summary_sha256=sha(summary_path), counts=summary['counts'],
                           eval_counts=summary.get('eval_counts', {}))
            except subprocess.TimeoutExpired:
                job['status'] = 'hard_timeout_child_killed_and_waited'
                raise
            finally:
                job.update(completed_utc=utc(), elapsed_seconds=time.perf_counter()-begin)
                write(output / 'stage.json', record)
        if summaries['dense']['counts'] != summaries['sparse']['counts'] or summaries['dense']['cursor'] != summaries['sparse']['cursor']:
            raise RuntimeError('D/E completion data exposure/update counters differ')
        record['status'] = 'scientific_pair_complete_quality_not_yet_adjudicated'
        return record
    except BaseException as error:
        if record is not None:
            record.update(status='stopped_with_failure', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if record is not None:
                record.update(completed_utc=utc(), elapsed_seconds=time.perf_counter()-started, estimated_stage_cost_usd=stage_cost())
                write(output / 'stage.json', record)
        finally:
            for lock in reversed(locks):
                lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = launch(args.protocol, args.protocol_sha256, args.output_dir, args.execute)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
