"""Bounded new-Pod numerical check and 12-update sparse engineering replay."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    raw = args.protocol.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError('Protocol hash mismatch')
    p = json.loads(raw)
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'steps': ['tiny_gpu_numerics', 'sparse_12_updates']}))
        return
    assert p['scope'] == 'gpu_preflight' and p['max_updates'] == 12
    assert p['device'] == 'cuda' and p['dtype'] == 'float32'
    assert p['launch_allowed'] is True and p['pod_id'] == os.environ.get('RUNPOD_POD_ID')
    assert 0 < p['hourly_rate_usd'] <= 1 and 0 <= p['stage_spent_usd'] < p['paid_ceiling_usd'] <= 10
    assert 0 < p['max_wall_seconds'] <= 1800
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Fresh output required')
    locks = []
    started = time.monotonic()
    record = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': args.sha256, 'scope': 'engineering_migration_only', 'jobs': []}
    def cost():
        return p['stage_spent_usd'] + (time.monotonic()-started)*p['hourly_rate_usd']/3600
    def run(label, command, seconds):
        limit = min(seconds, (p['paid_ceiling_usd']-cost()) / p['hourly_rate_usd']*3600-5)
        if limit <= 0:
            raise RuntimeError('Preflight cap exhausted')
        job = {'label': label, 'started_utc': datetime.now(timezone.utc).isoformat(), 'timeout_seconds': limit}
        record['jobs'].append(job)
        write(out/'stage.json', record)
        with (out/(label+'.stdout.log')).open('xb') as stdout, (out/(label+'.stderr.log')).open('xb') as stderr:
            done = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=limit)
        job.update(returncode=done.returncode, completed_utc=datetime.now(timezone.utc).isoformat())
        if done.returncode:
            raise RuntimeError(label+' failed')
    try:
        for name in ['babylm-scientific-pair.lock', 'babylm-gpu-preflight.lock']:
            lock = ROOT/'logs'/name
            with lock.open('x') as f:
                json.dump({'pid': os.getpid(), 'output': str(out), 'scope': record['scope']}, f)
            locks.append(lock)
        out.mkdir(parents=True, exist_ok=False)
        write(out/'input-protocol.json', p)
        run('gpu-numerics', [sys.executable, str(ROOT/'scripts/check_babylm_gpu_numerics_v0.py'), '--output', str(out/'gpu-numerics.json')], 300)
        numerical = json.loads((out/'gpu-numerics.json').read_text())
        assert numerical.get('passed') is True and numerical.get('status') == 'passed'
        q = dict(p, stage_spent_usd=cost())
        write(out/'sparse-protocol.json', q)
        run('sparse', [sys.executable, str(ROOT/'scripts/run_babylm_de_v0.py'), '--protocol', str(out/'sparse-protocol.json'), '--mode', 'sparse', '--output-dir', str(out/'sparse')], p['max_wall_seconds'])
        summary = json.loads((out/'sparse/summary.json').read_text())
        assert summary['status'] == 'max_updates_reached' and summary['counts']['engineering_updates'] == 12
        assert summary['counts']['scientific_updates'] == 0
        record['status'] = 'sparse_migration_preflight_complete'
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record.update(completed_utc=datetime.now(timezone.utc).isoformat(), estimated_pod_cost_usd=cost())
        if out.exists():
            write(out/'stage.json', record)
        for lock in reversed(locks):
            if json.loads(lock.read_text()).get('pid') == os.getpid():
                lock.unlink()


if __name__ == '__main__':
    main()
