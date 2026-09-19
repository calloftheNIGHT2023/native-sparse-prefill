"""Bounded sequential engineering preflight. Default is a plan, never a launch.

Does not rent/restart a Pod. The caller supplies a reconciled remaining budget,
actual hourly quote and prior setup spend. Child timeouts kill only this
launcher's child Python process; the current reference launches no workers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text(encoding='utf-8'))
    if p.get('scope') != 'gpu_preflight' or p.get('device') != 'cuda':
        raise ValueError('Only the bounded CUDA engineering protocol is accepted')
    if not 0 < p['max_updates'] <= 12 or not 0 < p['max_wall_seconds'] <= 1800:
        raise ValueError('Preflight cannot exceed 12 updates/group or 1800 seconds/group')
    if not args.execute:
        print(json.dumps({'status': 'plan_only_no_processes_started',
                          'stages': ['tiny_gpu_numerics', 'dense', 'sparse'],
                          'updates_per_group_max': p['max_updates'],
                          'launch_allowed_in_protocol': p.get('launch_allowed', False)}))
        return 0
    if p.get('launch_allowed') is not True or p.get('budget_reconciled') is not True:
        raise ValueError('Explicit protocol launch and reconciled budget are required')
    rate = float(p['hourly_rate_usd'])
    spent = float(p['stage_spent_usd'])
    ceiling = float(p['pair_stage_ceiling_usd'])
    remaining = float(p['remaining_research_budget_usd'])
    if not all(math.isfinite(v) for v in (rate, spent, ceiling, remaining)):
        raise ValueError('All quoted cost fields must be finite')
    if rate <= 0 or not 0 <= spent < ceiling <= 10 or remaining < ceiling - spent:
        raise ValueError('Preflight must fit the remaining research budget and USD10 total-stage cap')
    if float(p['paid_ceiling_usd']) != ceiling:
        raise ValueError('The engine and launcher must share one whole-stage ceiling')
    if p.get('eval_initial') or p.get('eval_final') or p.get('eval_every_updates'):
        raise ValueError('This short engineering preflight does not include a dev-evaluation batch')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory; no automatic repeat/resume of a preflight')
    output.mkdir(parents=True, exist_ok=True)
    lock = ROOT / 'logs/babylm-gpu-preflight.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'utc': utc(), 'output': str(output)}, stream)
    started = time.perf_counter()
    record = {'started_utc': utc(), 'scope': 'engineering_only', 'status': 'running',
              'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              'hourly_rate_usd': rate, 'stage_spent_usd_before': spent,
              'whole_stage_ceiling_usd': ceiling, 'jobs': [],
              'cost_scope': 'Quoted compute cost plus supplied setup spend; not an invoice. Storage/tax need reconciliation.'}

    def stage_cost():
        return spent + (time.perf_counter() - started) * rate / 3600

    def child(label, command, max_seconds):
        available_seconds = (ceiling - stage_cost()) / rate * 3600
        timeout = min(float(max_seconds), available_seconds - 5.0)
        if timeout < 1:
            raise RuntimeError('Stage budget exhausted before starting the next child')
        job = {'label': label, 'started_utc': utc(), 'hard_timeout_seconds': timeout}
        record['jobs'].append(job)
        write(output / 'stage.json', record)
        begin = time.perf_counter()
        try:
            with (output / f'{label}.stdout.log').open('wb') as stdout, (output / f'{label}.stderr.log').open('wb') as stderr:
                completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                           timeout=timeout, check=False)
            job['returncode'] = completed.returncode
            if completed.returncode:
                raise RuntimeError(f'{label} failed; inspect saved stdout/stderr')
        except subprocess.TimeoutExpired:
            job['status'] = 'hard_timeout_child_killed_and_waited'
            raise
        finally:
            job.update(completed_utc=utc(), elapsed_seconds=time.perf_counter() - begin)
            write(output / 'stage.json', record)

    try:
        child('gpu-numerics', [sys.executable, str(ROOT / 'scripts/check_babylm_gpu_numerics_v0.py'),
                              '--output', str(output / 'gpu-numerics.json')], 300)
        numerical = json.loads((output / 'gpu-numerics.json').read_text(encoding='utf-8'))
        if numerical.get('passed') is not True:
            raise RuntimeError('GPU numerical gate did not explicitly pass')
        for mode in ('dense', 'sparse'):
            run_protocol = dict(p, stage_spent_usd=stage_cost())
            protocol_path = output / f'{mode}-protocol.json'
            write(protocol_path, run_protocol)
            child(mode, [sys.executable, str(ROOT / 'scripts/run_babylm_de_v0.py'),
                         '--protocol', str(protocol_path), '--mode', mode,
                         '--output-dir', str(output / mode)], p['max_wall_seconds'])
            summary = json.loads((output / mode / 'summary.json').read_text(encoding='utf-8'))
            if summary.get('status') != 'max_updates_reached':
                raise RuntimeError(f'{mode} did not complete the planned bounded preflight; stop pair')
        record['status'] = 'paired_preflight_complete_not_scientific_training'
        return 0
    except BaseException as error:
        record.update(status='stopped_with_failure', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record.update(completed_utc=utc(), elapsed_seconds=time.perf_counter() - started,
                      estimated_stage_cost_usd=stage_cost())
        try:
            write(output / 'stage.json', record)
        finally:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
