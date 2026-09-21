"""One finite, fresh D -> E sequence. No retry, resume, or follow-on stage."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_babylm_one_epoch_v0 import load, resolve, require, sha, validate, write


def paired_audit(out):
    evidence = {}
    for mode in ('dense', 'sparse'):
        run = out / mode / 'run'
        summary = load(run / 'summary.json')
        tail = load(out / mode / 'tail100-audit.json')
        require(summary['status'] == 'epoch_complete' and tail['status'] == 'audited_epoch_complete',
                'Both groups must finish and pass the one-epoch audit')
        with (run / 'events.jsonl').open() as f:
            events = [json.loads(line) for line in f]
        evidence[mode] = (summary, tail, events[0], [e for e in events if e['type'] == 'update'])
    d, e = evidence['dense'], evidence['sparse']
    for field in ('protocol_sha256', 'source_hashes', 'data_fingerprint', 'counts', 'last_eval_point'):
        require(d[0][field] == e[0][field], 'Paired metadata mismatch: ' + field)
    initial_d, initial_e = d[2]['initial_parameter_hashes'], e[2]['initial_parameter_hashes']
    require(initial_d and set(initial_d).issubset(initial_e), 'Shared initial parameters missing')
    require(all(initial_d[name] == initial_e[name] for name in initial_d), 'Shared random backbone differs')
    require(len(d[3]) == len(e[3]) == 1413, 'Paired full epoch required')
    for a, b in zip(d[3], e[3]):
        for field in ('window_ids', 'counts', 'cursor', 'loss_tokens_this_update', 'lr_word_position', 'lr_multiplier'):
            require(a[field] == b[field], 'Paired update mismatch: ' + field)
        require(a['lr']['backbone'] == b['lr']['backbone'], 'Paired backbone LR mismatch')
        require(a['gradient_clip_scope'] == b['gradient_clip_scope'] == 'separate_backbone_indexer',
                'Clipping scope differs')
    result = {'status': 'paired_epoch_complete_audited', 'paired_updates': len(d[3]),
              'shared_initial_backbone_exact': True, 'data_order_and_backbone_lr_exact': True,
              'tail100': {mode: row[1]['tail'] for mode, row in evidence.items()},
              'delta_sparse_minus_dense': {key: e[1]['tail']['metrics'][key] - d[1]['tail']['metrics'][key]
                  for key in ('token_weighted_lm_nll', 'token_weighted_lm_ppl',
                              'arithmetic_mean_step_lm_nll', 'arithmetic_mean_step_lm_ppl')},
              'raw_evidence_sha256': {str(path.relative_to(out)): sha(path)
                  for mode in ('dense', 'sparse') for path in
                  (out / mode / 'run' / 'summary.json', out / mode / 'run' / 'events.jsonl',
                   out / mode / 'tail100-audit.json')},
              'scope': 'One seed, online training tail; not held-out quality, warmup mechanism, speed or novelty evidence',
              'new_model_calls': 0, 'completed_utc': datetime.now(timezone.utc).isoformat()}
    write(out / 'paired-tail100-audit.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True, type=Path)
    ap.add_argument('--protocol-sha256', required=True)
    ap.add_argument('--numerics', required=True, type=Path)
    ap.add_argument('--numerics-sha256', required=True)
    ap.add_argument('--output-dir', required=True, type=Path)
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    pp, out = resolve(a.protocol), resolve(a.output_dir)
    p = load(pp)
    require(sha(pp) == a.protocol_sha256, 'Protocol hash mismatch')
    require(p['source_sha256'].get('scripts/run_babylm_one_epoch_sequential_v0.py') == sha(Path(__file__)),
            'Sequential dispatcher must be pinned')
    plan = validate(p)
    require(sha(resolve(a.numerics)) == a.numerics_sha256, 'Numerics receipt hash mismatch')
    if not a.execute:
        print(json.dumps({'status': 'plan_only', 'order': ['dense', 'sparse'], 'plan': plan}))
        return 0
    require(out.is_relative_to(ROOT / 'results') and not out.exists(), 'Require new sequence output')
    out.mkdir(parents=True, exist_ok=False)
    stage = {'status': 'starting', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
             'protocol_sha256': a.protocol_sha256, 'order': ['dense', 'sparse'], 'completed_modes': [],
             'policy': 'E only after audited D completion; no automatic retry/resume/additional run'}
    write(out / 'sequence-stage.json', stage)
    started = time.monotonic()
    try:
        for mode in ('dense', 'sparse'):
            stage.update(status='running', active_mode=mode)
            write(out / 'sequence-stage.json', stage)
            cmd = [sys.executable, '-u', str(ROOT / 'scripts/run_babylm_one_epoch_mig_worker_v0.py'),
                   '--protocol', str(pp), '--protocol-sha256', a.protocol_sha256,
                   '--numerics', str(resolve(a.numerics)), '--numerics-sha256', a.numerics_sha256,
                   '--mode', mode, '--output-dir', str(out / mode), '--execute']
            with (out / (mode + '-worker.log')).open('wb') as log:
                child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                stage['worker_pid'] = child.pid
                write(out / 'sequence-stage.json', stage)
                code = child.wait()
            require(code == 0, mode + ' worker failed; E/retry is not permitted')
            require(load(out / mode / 'tail100-audit.json')['status'] == 'audited_epoch_complete',
                    'Missing completed tail100 audit')
            stage['completed_modes'].append(mode)
            write(out / 'sequence-stage.json', stage)
        paired_audit(out)
        stage.update(status='complete_paired_audited', active_mode=None)
        return 0
    except BaseException as error:
        stage.update(status='failed_or_incomplete', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stage.update(updated_utc=datetime.now(timezone.utc).isoformat(), elapsed_wall_seconds=time.monotonic()-started)
        write(out / 'sequence-stage.json', stage)


if __name__ == '__main__':
    raise SystemExit(main())
