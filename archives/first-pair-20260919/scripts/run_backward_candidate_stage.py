"""Bounded stage controller: wait for current build, gate, time, then integrate.

Every dependent experiment has a correctness gate. No retries or long training.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/flashmoba-backward-stage-controller-v0'
ENV = Path('/opt/native-sparse-flashmoba-env-v2')
PY = ENV/'bin/python'
CAND = ROOT/'third_party/flash-moba-barrier-candidate-v0'
BUILD = ROOT/'results/flashmoba-backward-environment-v5'


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    started = utc()
    def event(kind, **kw):
        row = dict(utc=utc(), event=kind, **kw)
        with (OUT/'events.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    def run(label, command, candidate=False, timeout=300, cwd=ROOT, env=None):
        if env is None:
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            if candidate:
                env['PYTHONPATH'] = str(CAND)
        event('command_start', label=label, candidate=candidate, command=[str(x) for x in command])
        tick = time.perf_counter()
        with (OUT/f'{label}.log').open('w') as f:
            p = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                               stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        event('command_end', label=label, returncode=p.returncode, seconds=time.perf_counter()-tick)
        if p.returncode:
            raise RuntimeError(f'{label} returned {p.returncode}: '+(OUT/f'{label}.log').read_text()[-2000:])
    result = dict(status='running', started_utc=started)
    try:
        event('waiting_for_existing_build', directory=str(BUILD))
        deadline = time.monotonic()+2300
        while not (BUILD/'environment.json').exists():
            if time.monotonic()>deadline:
                raise TimeoutError('Existing build did not finish within the bounded wait.')
            time.sleep(5)
        build = json.loads((BUILD/'environment.json').read_text())
        assert build['status']=='complete', build
        event('build_ready', extension_sha256=build['candidate_extension_sha256'])
        run('long-gate', [PY, 'scripts/verify_long_backward_candidate.py', '--output',
                         'results/flashmoba-long-backward-barrier-v0', '--label', 'barrier_candidate', '--repeats', '16'], candidate=True)
        gate = json.loads((ROOT/'results/flashmoba-long-backward-barrier-v0/result.json').read_text())
        assert gate['extension_sha256']==build['candidate_extension_sha256']
        assert gate['extension_sha256']!='b114a7755aad6556bc72eac1f8de0fcd0d5bd1e78d0dc1b1e521ac8621ce4d53'
        assert gate['actual_k4_selected_masks_match'] and gate['deterministic_repeat_pass'] and gate['reference_tolerance_pass'], 'Long backward candidate correctness gate failed.'
        run('standard-gate', [PY, 'scripts/verify_flashmoba_official_v1.py', '--output',
                             'results/flashmoba-newpod-standard-barrier-v0'], candidate=True)
        standard = json.loads((ROOT/'results/flashmoba-newpod-standard-barrier-v0/verification.json').read_text())
        assert standard['status']=='passed' and len(standard['checks'])==12
        event('all_kernel_gates_passed')
        for i, candidate in enumerate([False, True, True, False]):
            label = 'barrier' if candidate else 'original'
            run(f'timing-{i}-{label}', [PY, 'scripts/benchmark_backward_barrier.py', '--output',
                f'results/flashmoba-backward-timing-{i}-{label}-v0', '--label', label], candidate=candidate)
        run('training-integration', [PY, 'scripts/run_backward_barrier_training.py', '--data',
            'data/flashmoba-backward-training-v0', '--output', 'results/flashmoba-backward-training-v0'], candidate=True, timeout=900)
        train = json.loads((ROOT/'results/flashmoba-backward-training-v0/result.json').read_text())
        assert train['status']=='complete' and train['scientific_optimizer_updates']==16
        train_pass = all(p['first_gradient']['changed_elements']==0 and
                         p['final_parameter_delta']['changed_elements']==0 and
                         p['first_forward_changed_route_rows']==0 for p in train['pairs'])
        result.update(status='complete', kernel_gates_passed=True, integration_repeat_pass=train_pass,
                      scientific_optimizer_updates=train['scientific_optimizer_updates'])
        event('stage_complete', **result)
    except Exception:
        result.update(status='blocked_at_gate_or_execution', error=traceback.format_exc())
        event('stage_stopped', **result)
    finally:
        result['finished_utc'] = utc()
        (OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    if result['status']!='complete':
        raise SystemExit(1)


if __name__=='__main__':
    main()
