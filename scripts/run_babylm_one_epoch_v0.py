"""Frozen one-pass D/E worker. Default: validate/plan only, no model calls.

Execution needs an explicit protocol hash and an original-threshold numerical
receipt for the current GPU/runtime. No automatic resume, retries or next run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve(path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def validate(protocol):
    import numpy as np
    p = protocol
    require(p['max_epochs'] == 1 and p['expected_stop_reason'] == 'epoch_complete', 'Requires exactly one epoch')
    require(p['training_initialization'] == 'fresh_random_shared_backbone_no_resume', 'Fresh initialization required')
    require(p['gradient_clip_scope'] == 'separate_backbone_indexer', 'Independent clipping required')
    require(p['tail_metric_updates'] == 100, 'Tail metric must be fixed before training')
    require(bool(p['source_sha256']), 'Code pins are not frozen')
    for name, digest in p['source_sha256'].items():
        path = resolve(name)
        require(path.is_relative_to(ROOT) and sha(path) == digest, 'Source pin mismatch: ' + name)
    manifest_path = resolve(p['train_manifest'])
    require(sha(manifest_path) == p['train_manifest_sha256'], 'Training manifest changed')
    m = load(manifest_path)
    for entry in m['artifacts']:
        path = (manifest_path.parent / entry['path']).resolve()
        require(path.parent == manifest_path.parent and sha(path) == entry['sha256'], 'Window artifact changed')
    index = np.load(manifest_path.parent / 'windows.u64.npy', mmap_mode='r', allow_pickle=False)
    columns = {name: i for i, name in enumerate(m['columns'])}
    expected = p['expected_epoch_counts']
    require(len(index) == expected['windows'], 'Window count mismatch')
    require(bool(np.all(index[:, columns['next_token_loss_positions']] > 0)), 'Unexpected zero-target windows')
    order = np.random.default_rng(np.random.SeedSequence([p['data_order_seed'], 0])).permutation(len(index))
    batches = [order[i:i + p['windows_per_update']] for i in range(0, len(order), p['windows_per_update'])]
    require(len(batches) == expected['updates'] == p['max_updates'], 'Update count mismatch')
    require(len(batches[-1]) == expected['final_update_windows'], 'Partial final batch mismatch')
    for key, col in [('word_exposures', 'word_exposures'), ('input_tokens', 'input_tokens'),
                     ('loss_tokens', 'next_token_loss_positions')]:
        require(int(index[:, columns[col]].sum()) == expected[key], 'Epoch ledger mismatch: ' + key)
    require(p['max_word_exposures'] == expected['word_exposures'], 'LR horizon is not the complete epoch')
    require(p['warmup_word_exposures'] < p['max_word_exposures'], 'Invalid one-epoch warmup')
    tail = np.concatenate(batches[-100:])
    return {'mode': 'validation_only_no_model_calls', 'expected_counts': expected,
            'permutation_sha256': hashlib.sha256(order.astype('<u8').tobytes()).hexdigest(),
            'tail_update_range': [len(batches) - 99, len(batches)],
            'tail_loss_tokens': int(index[tail, columns['next_token_loss_positions']].sum()),
            'tail_word_exposures': int(index[tail, columns['word_exposures']].sum()),
            'final_backbone_lr': p['learning_rate'] * p['min_lr_ratio'],
            'final_indexer_lr': p['indexer_learning_rate'] * p['min_lr_ratio']}


def numerical_gate(path, digest):
    from scripts.run_babylm_sparse_parallel_v0 import verify_numerics, portable_hashes
    pod_id = os.environ.get('RUNPOD_POD_ID')
    require(bool(pod_id), 'RunPod worker identity is missing')
    report = verify_numerics({'numerics_path': str(path), 'numerics_sha256': digest, 'pod_id': pod_id})
    import importlib.util
    import torch
    import numpy as np
    for name, expected in portable_hashes(report['source_hashes']).items():
        if name.startswith('transformers/'):
            spec = importlib.util.find_spec('transformers')
            actual = Path(spec.origin).parent / name.removeprefix('transformers/')
        else:
            actual = ROOT / name
        require(sha(actual) == expected, 'Numerical receipt source mismatch: ' + name)
    require(torch.__version__ == report['runtime']['torch'] and np.__version__ == report['runtime']['numpy']
            and torch.version.cuda == report['runtime']['cuda_compiled_version'], 'Numerical receipt runtime mismatch')
    cmd = ['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total', '--format=csv,noheader']
    smi = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=20).stdout.strip()
    require(smi == report['hardware']['nvidia_smi']['stdout'].strip(), 'GPU changed: needs its own numerical receipt')
    rows = list(csv.reader(io.StringIO(smi)))
    require(len(rows) == 1, 'Worker expects exactly one visible GPU')
    return rows[0][2].strip(), report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--mode', choices=['dense', 'sparse'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--numerics', type=Path)
    parser.add_argument('--numerics-sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    pp = resolve(args.protocol)
    p = load(pp)
    plan = validate(p)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    require(os.name == 'posix', 'Scientific worker requires the validated Linux GPU environment')
    require(args.protocol_sha256 and sha(pp) == args.protocol_sha256, 'Explicit frozen protocol SHA required')
    require(args.numerics is not None and args.numerics_sha256, 'Numerical receipt and SHA required')
    gpu, report = numerical_gate(resolve(args.numerics), args.numerics_sha256)
    import fcntl
    lock = open('/tmp/babylm-one-epoch-' + gpu + '.lock', 'a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Check once after owning our lock. Never stop someone else's GPU process.
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True, timeout=20).stdout.strip()
    require(not active, 'GPU has active compute processes; refusing duplicate launch')
    out = resolve(args.output_dir)
    require(out.is_relative_to(ROOT / 'results') and not out.exists(), 'Use a new results directory')
    out.mkdir(parents=True, exist_ok=False)
    stage = {'status': 'starting', 'mode': args.mode, 'created_utc': datetime.now(timezone.utc).isoformat(),
             'pid': os.getpid(), 'gpu_uuid': gpu, 'protocol_sha256': sha(pp), 'plan': plan,
             'numerics_sha256': args.numerics_sha256, 'new_numerics_calls': 0,
             'no_automatic_resume_or_followup': True}
    write(out / 'worker-stage.json', stage)
    command = [sys.executable, '-u', str(ROOT / 'scripts/run_babylm_de_v0.py'),
               '--protocol', str(pp), '--mode', args.mode, '--output-dir', str(out / 'run')]
    started = time.monotonic()
    try:
        with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
            stage.update(status='running', child_pid=child.pid)
            write(out / 'worker-stage.json', stage)
            try:
                code = child.wait(timeout=p['hard_timeout_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise RuntimeError('Hard timeout: retain partial evidence, do not auto-resume')
        require(code == 0, 'Training process failed; inspect preserved stderr and failure checkpoint')
        subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_babylm_one_epoch_v0.py'),
                        '--run-dir', str(out / 'run'), '--output', str(out / 'tail100-audit.json')],
                       check=True, cwd=ROOT, timeout=120)
        stage.update(status='complete_one_epoch_audited', child_returncode=code)
        return 0
    except BaseException as error:
        stage.update(status='failed_or_incomplete', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stage.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_wall_seconds=time.monotonic()-started)
        stage['estimated_stage_cost_usd'] = stage['elapsed_wall_seconds'] / 3600 * p['hourly_rate_usd']
        stage['cost_basis'] = 'Conservative worker wall estimate; excludes idle before launch, not invoice'
        write(out / 'worker-stage.json', stage)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
