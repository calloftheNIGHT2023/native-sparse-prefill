"""Finite, source-pinned checkpoint evaluation queue; never dispatches training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def replay_audit(expected, summary, rows, tolerance):
    """Validate fixed panel counts and NLL, never adjust tolerance on failure."""
    actual = summary['total']
    reference = expected['total']
    for key in ('windows', 'forward_calls', 'input_tokens', 'loss_tokens', 'word_exposures'):
        require(actual[key] == reference[key], f'Replay counter differs: {key}')
    delta = abs(actual['nll'] - reference['nll'])
    require(math.isfinite(delta) and delta <= tolerance['aggregate_nll_abs'], 'Replay aggregate NLL differs')
    require(actual['attention_counts'] == reference['attention_counts'], 'Replay attention counts differ')
    old_rows = expected['position_diagnostics']['per_window']
    require(len(rows) == len(old_rows), 'Replay row count differs')
    require([r['window_index'] for r in rows] == [r['window_index'] for r in old_rows], 'Replay row order differs')
    errors = []
    for new, old in zip(rows, old_rows):
        require(new['loss_tokens'] == old['loss_tokens'], 'Replay row target count differs')
        if old['loss_tokens']:
            error = abs(new['nll_sum'] / new['loss_tokens'] - old['nll_sum'] / old['loss_tokens'])
            require(math.isfinite(error) and error <= tolerance['window_nll_abs'], 'Replay per-window NLL differs')
            errors.append(error)
    return {'passed': True, 'aggregate_nll_abs': delta, 'max_window_nll_abs': max(errors, default=0),
            'counts_exact': True, 'attention_counts_exact': True, 'tolerance': tolerance}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_jobs(jobs, protocols):
    require(4 <= len(jobs) <= 16, 'Expected two replays, two full-dev jobs, at most twelve diagnostics')
    require(len({j['name'] for j in jobs}) == len(jobs), 'Job names must be unique')
    require([j['role'] for j in jobs[:2]] == ['replay', 'replay'], 'Both replays must precede other evaluations')
    require([p['mode'] for p in protocols[:2]] == ['dense', 'sparse'], 'Replay order must be dense then sparse')
    require(sum(j['role'] == 'replay' for j in jobs) == 2, 'Exactly two replays required')
    require(sum(j['role'] == 'full_dev' for j in jobs) == 2, 'Exactly two full-dev jobs required')
    identities = {p['mode']: p['checkpoint_sha256'] for p in protocols[:2]}
    full_modes = []
    for job, p in zip(jobs, protocols):
        require(job['role'] in ('replay', 'diagnostic', 'full_dev'), 'Unknown job role')
        require(p['dtype'] == 'float32' and p['device'] == 'cuda', 'Frozen CUDA FP32 evaluation only')
        if job['role'] == 'replay':
            require('replay_reference' in job and len(p['window_indices']) == 48, 'Pinned 48-window replay required')
            require(not p.get('enable_routing_diagnostics', False), 'Replay must use uninstrumented original path')
        elif job['role'] == 'full_dev':
            require(p.get('window_indices') is None, 'Full dev must include every window')
            require(p['checkpoint_sha256'] == identities[p['mode']], 'Full-dev checkpoint must match its replay')
            require(not p.get('enable_routing_diagnostics', False), 'Full dev must use original routing')
            full_modes.append(p['mode'])
        else:
            require(p['mode'] == 'sparse' and p.get('enable_routing_diagnostics') is True,
                    'Diagnostics require explicit sparse instrumentation')
            require(len(p['window_indices']) == 48, 'Diagnostics use the frozen 48-window panel')
    require(sorted(full_modes) == ['dense', 'sparse'], 'One full-dev job per mode required')


def run(protocol_path, protocol_sha):
    require(os.name == 'posix', 'This finite cloud controller requires Linux')
    pp = (ROOT / protocol_path).resolve()
    require(sha(pp) == protocol_sha, 'Master protocol hash differs')
    p = load_json(pp)
    require(p['schema_version'] == 1 and p['launch_allowed'] is True, 'Unfrozen master')
    require(0 < p['max_wall_seconds'] <= 14280 and p['hard_timeout_seconds'] == 14400,
            'Reserve two minutes within the four-hour hard limit')
    require(0 < p['hourly_rate_usd'] <= 0.81 and p['stage_cost_cap_usd'] <= 3.24, 'Stage cost exceeds frozen bound')
    require(p['max_wall_seconds'] / 3600 * p['hourly_rate_usd'] <= p['stage_cost_cap_usd'] + 1e-9, 'Inconsistent cost cap')
    for filename, expected in p['source_sha256'].items():
        require(sha(ROOT / filename) == expected, 'Source differs: ' + filename)
    require(p['source_sha256'].get('scripts/run_babylm_optimization_stage_a_v0.py') == sha(__file__), 'Controller unpinned')
    protocols = []
    for job in p['jobs']:
        child_path = ROOT / job['protocol']
        require(sha(child_path) == job['protocol_sha256'], 'Child protocol changed')
        protocols.append(load_json(child_path))
    validate_jobs(p['jobs'], protocols)
    out = (ROOT / p['output_dir']).resolve()
    require(out.is_relative_to(ROOT / 'results') and not out.exists(), 'Fresh results output required')
    import fcntl
    from scripts.run_babylm_one_epoch_mig_worker_v0 import isolation_gate
    lock_path = '/tmp/babylm-one-epoch-' + p['execution_hardware']['mig_uuid'] + '.lock'
    with open(lock_path, 'a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(parents=True, exist_ok=False)
        state = {'status': 'preparing', 'pid': os.getpid(), 'started_utc': utc(), 'protocol_sha256': protocol_sha,
                 'jobs': [], 'training_updates': 0, 'backwards': 0, 'automatic_monitoring_resumed': False,
                 'lock_path': lock_path, 'new_numerical_preflight_model_calls': 0}
        start = time.monotonic()
        child = None
        child_record = None
        replay_passed = set()
        def stopped(signum, frame):
            raise InterruptedError(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, stopped)
        write(out / 'stage.json', state)
        try:
            state['isolation'] = isolation_gate(pp, p['execution_hardware'])
            for job in p['jobs']:
                remaining = p['max_wall_seconds'] - (time.monotonic() - start)
                require(remaining > 60, 'Stage time budget exhausted; no new job')
                path = ROOT / job['protocol']
                require(sha(path) == job['protocol_sha256'], 'Child protocol changed')
                jp = load_json(path)
                if job['role'] != 'replay':
                    require(replay_passed == {'dense', 'sparse'}, 'Both replays must pass before new evaluation')
                job_out = (ROOT / jp['output_dir']).resolve()
                require(job_out.is_relative_to(out) and not job_out.exists(), 'Child output must be fresh under stage root')
                child_record = {'name': job['name'], 'started_utc': utc(), 'status': 'running', 'output_dir': jp['output_dir']}
                state['jobs'].append(child_record)
                state['status'] = 'running'
                with (out / (job['name'] + '.stdout.log')).open('xb') as stdout, (out / (job['name'] + '.stderr.log')).open('xb') as stderr:
                    # Same process group as this controller: the external timeout
                    # can kill the entire finite queue, including its active evaluator.
                    child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/run_babylm_checkpoint_eval_v0.py'),
                                              '--protocol', str(path)], cwd=ROOT, stdout=stdout, stderr=stderr)
                    child_record['pid'] = child.pid
                    write(out / 'stage.json', state)
                    code = child.wait(timeout=min(remaining, jp['max_wall_seconds'] + 60))
                child_record.update(returncode=code, finished_utc=utc())
                require(code == 0, 'Evaluator failed; no automatic retry: ' + job['name'])
                summary = load_json(job_out / 'summary.json')
                counts = summary.get('counts', summary)
                require(counts.get('optimizer_updates') == 0, 'Unexpected optimizer updates')
                require(summary['status'] == 'evaluation_complete', 'Evaluator did not finish')
                require(counts.get('backward_calls') == 0, 'Unexpected backwards')
                manifest = load_json(ROOT / jp['dev_manifest'])
                expected_ids = list(range(manifest['total_windows'])) if jp.get('window_indices') is None else jp['window_indices']
                rows = [json.loads(line) for line in (job_out / 'windows.jsonl').read_text().splitlines()]
                require([r['window_index'] for r in rows] == expected_ids, 'Observed evaluation IDs differ from frozen job')
                require(summary['total']['windows'] == summary['total']['forward_calls'] == len(expected_ids), 'Evaluation count differs')
                require(math.isfinite(summary['total']['nll']), 'Nonfinite final NLL')
                if job['role'] == 'full_dev':
                    for key, mk in [('input_tokens', 'input_tokens_per_pass'), ('loss_tokens', 'next_token_loss_positions_per_pass'),
                                    ('word_exposures', 'word_exposures_per_pass')]:
                        require(summary['total'][key] == manifest[mk], 'Full-dev accounting mismatch: ' + key)
                if 'replay_reference' in job:
                    reference_path = ROOT / job['replay_reference']
                    require(sha(reference_path) == job['replay_reference_sha256'], 'Replay reference changed')
                    audit = replay_audit(load_json(reference_path), summary, rows, p['replay_tolerance'])
                    write(job_out / 'replay-audit.json', audit)
                    child_record['replay_passed'] = True
                    replay_passed.add(jp['mode'])
                child_record.update(status='complete', total=summary['total'])
                child = None
                write(out / 'stage.json', state)
            state['status'] = 'complete'
            return 0
        except BaseException as error:
            state.update(status='failed_or_incomplete', error_type=type(error).__name__, error=str(error))
            if child_record is not None and child_record['status'] == 'running':
                child_record.update(status='failed_or_incomplete', error_type=type(error).__name__)
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if child is not None and child_record is not None:
                child_record['observed_final_returncode'] = child.poll()
            state.update(finished_or_updated_utc=utc(), elapsed_seconds=time.monotonic() - start)
            state['estimated_stage_cost_usd'] = state['elapsed_seconds'] / 3600 * p['hourly_rate_usd']
            state['cost_basis'] = 'Controller wall time only, includes child stages; not invoice, excludes prelaunch idle'
            write(out / 'stage.json', state)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.protocol, args.protocol_sha256))
