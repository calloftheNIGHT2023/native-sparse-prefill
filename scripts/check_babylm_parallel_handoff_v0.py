"""Revoke only the queued E dispatch by moving the old pair's master protocol.

Plan-only by default. --revoke is Linux-only and sends no signals. This does
not launch E, stop D, alter its child protocol, modify the old stage, or claim
that the pair completed. A separate receipt explains the expected old-parent
FileNotFoundError as administrative dispatch revocation. Never restore the
master path until the old parent has exited: doing so could re-enable E.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

AUDITED_SOURCE_HASHES = {
    'scripts/run_babylm_scientific_pair_v0.py': '593280477ca8457ad21f171c7a6cec137e7715029647f78db1ea75bb28fbafe7',
    'scripts/run_babylm_de_v0.py': '0c463cd3079977ce74f2f6464c2e0497ec60a7d0a7ce376ae8368c7fd7d533f5',
    'src/babylm_hybrid/training.py': '2136e0b470d5b055418394e64451ddffb1f58f5c2fcd131ba1e01f0224cb494f',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def process_identity(pid):
    """Read Linux procfs only; start ticks protect against PID reuse."""
    if sys.platform != 'linux':
        raise RuntimeError('Actual handoff requires Linux procfs; no signal fallback')
    base = Path('/proc') / str(int(pid))
    stat = (base / 'stat').read_text().rsplit(') ', 1)[1].split()
    result = {'pid': int(pid), 'state': stat[0], 'ppid': int(stat[1]),
              'start_ticks': int(stat[19]), 'cwd': str((base / 'cwd').resolve()),
              'argv': [x.decode() for x in (base / 'cmdline').read_bytes().split(b'\0') if x]}
    # A process disappearing/reusing its PID during cmdline reads must fail.
    again = (base / 'stat').read_text().rsplit(') ', 1)[1].split()
    if int(again[19]) != result['start_ticks']:
        raise RuntimeError('PID identity changed while reading procfs')
    result['state'] = again[0]
    return result


def option(argv, flag):
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise ValueError('Expected exactly one process argument: ' + flag)
    return argv[argv.index(flag) + 1]


def path_arg(identity, value):
    path = Path(value)
    return (path if path.is_absolute() else Path(identity['cwd']) / path).resolve()


def alive(identity):
    if identity['state'] in ('Z', 'X', 'x') or not identity.get('argv'):
        raise RuntimeError('Expected a live non-zombie process')


def assert_same(before, after):
    alive(after)
    for key in ('pid', 'start_ticks', 'ppid', 'argv', 'cwd'):
        if before[key] != after[key]:
            raise RuntimeError('Process identity changed: ' + key)


def inspect_handoff(project_root, protocol_path, expected_sha256, pair_output,
                    controller_pid, dense_pid, process_reader=None):
    reader = process_reader or process_identity
    if Path(protocol_path).is_symlink():
        raise ValueError('The master protocol must not be a symbolic link')
    root, master, output = map(lambda x: Path(x).resolve(), (project_root, protocol_path, pair_output))
    for name, expected in AUDITED_SOURCE_HASHES.items():
        if sha(root / name) != expected:
            raise ValueError('Unaudited launcher/child source: ' + name)
    child = output / 'dense-protocol.json'
    input_copy = output / 'input-protocol.json'
    for path in (master, child, input_copy):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Expected distinct regular protocol files')
    if os.path.samefile(master, child) or os.path.samefile(master, input_copy):
        raise ValueError('Master and child/input copy must not alias')
    if sha(master) != expected_sha256 or sha(input_copy) != expected_sha256:
        raise ValueError('Master SHA or independently saved input copy mismatch')
    stage = load(output / 'stage.json')
    jobs = stage.get('jobs', [])
    if (stage.get('status') != 'running_dense' or stage.get('protocol_sha256') != expected_sha256 or
            len(jobs) != 1 or jobs[0].get('label') != 'dense' or
            'returncode' in jobs[0] or 'completed_utc' in jobs[0]):
        raise ValueError('Pair must still be waiting for its only dense child')
    if (output / 'sparse-protocol.json').exists() or (output / 'sparse').exists():
        raise ValueError('Sparse dispatch evidence already exists')
    if sha(child) != jobs[0].get('child_protocol_file_sha256'):
        raise ValueError('Dense child protocol SHA mismatch')
    dense, parent = reader(dense_pid), reader(controller_pid)
    alive(dense); alive(parent)
    if dense['ppid'] != controller_pid:
        raise ValueError('Dense process is not the controller direct child')
    for identity, script in ((dense, 'run_babylm_de_v0.py'), (parent, 'run_babylm_scientific_pair_v0.py')):
        if sum(path_arg(identity, arg) == (root / 'scripts' / script).resolve()
               for arg in identity['argv'] if not arg.startswith('-')) != 1:
            raise ValueError('Process is not the audited direct Python entrypoint')
    if (option(dense['argv'], '--mode') != 'dense' or
            path_arg(dense, option(dense['argv'], '--protocol')) != child.resolve() or
            path_arg(dense, option(dense['argv'], '--output-dir')) != (output / 'dense').resolve() or
            path_arg(parent, option(parent['argv'], '--protocol')) != master.resolve() or
            option(parent['argv'], '--protocol-sha256') != expected_sha256 or
            path_arg(parent, option(parent['argv'], '--output-dir')) != output.resolve() or
            '--execute' not in parent['argv']):
        raise ValueError('Process protocol/mode/output binding mismatch')
    return {'status': 'handoff_plan_no_changes', 'utc': utc(), 'master_path': str(master),
            'master_sha256': expected_sha256, 'pair_output': str(output),
            'dense_protocol_path': str(child), 'dense_protocol_sha256': sha(child),
            'input_protocol_sha256': sha(input_copy), 'controller_identity': parent,
            'dense_identity': dense, 'audited_source_hashes': AUDITED_SOURCE_HASHES,
            'dense_signals_sent': 0, 'launches_started': 0,
            'restriction': 'Do not restore the master path until the old controller exits.'}


def revoke_dispatch(project_root, protocol_path, expected_sha256, pair_output,
                    controller_pid, dense_pid, evidence_dir, process_reader=None):
    reader = process_reader or process_identity
    plan = inspect_handoff(project_root, protocol_path, expected_sha256, pair_output,
                           controller_pid, dense_pid, reader)
    evidence = Path(evidence_dir).absolute()
    evidence.mkdir(parents=True, exist_ok=False)
    master = Path(plan['master_path'])
    moved = evidence / 'revoked-master-protocol.json'
    receipt = dict(plan, status='administrative_dispatch_revocation_intent',
                   preserved_master_path=str(moved), moved=False,
                   revocation_confirmed=False)
    write_new(evidence / 'intent.json', receipt)
    try:
        if master.stat().st_dev != evidence.stat().st_dev:
            raise ValueError('Atomic move must remain on the same filesystem')
        assert_same(plan['dense_identity'], reader(dense_pid))
        assert_same(plan['controller_identity'], reader(controller_pid))
        # The fresh private evidence directory has no destination to overwrite.
        # Never substitute copy/unlink: removing the dispatch path is atomic.
        os.rename(master, moved)
        receipt['moved'] = True
        # This check is decisive: while the identical D child is still alive,
        # the parent cannot have returned from subprocess.run and passed E's
        # next master-file gate. If it exits in this interval, fail ambiguous.
        assert_same(plan['dense_identity'], reader(dense_pid))
        assert_same(plan['controller_identity'], reader(controller_pid))
        if master.exists() or sha(moved) != expected_sha256:
            raise RuntimeError('Revoked path was recreated or original bytes changed')
        if sha(plan['dense_protocol_path']) != plan['dense_protocol_sha256']:
            raise RuntimeError('Dense child protocol changed')
        if sha(Path(pair_output) / 'input-protocol.json') != expected_sha256:
            raise RuntimeError('Independent input protocol changed')
        receipt.update(status='administrative_dispatch_revoked_dense_still_running',
                       revocation_confirmed=True, completed_utc=utc(),
                       expected_old_parent_status='stopped_with_failure',
                       expected_old_parent_error='FileNotFoundError at next master-protocol gate',
                       interpretation='Expected dispatch revocation, not a dense training failure. Verify the final dense summary separately.')
        write_new(evidence / 'revocation.json', receipt)
        return receipt
    except BaseException as error:
        receipt.update(status='handoff_failed_or_raced_do_not_launch_second_sparse',
                       revocation_confirmed=False, completed_utc=utc(), error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
        write_new(evidence / 'failure.json', receipt)
        # Do not restore the master and accidentally re-authorize old E.
        raise


def verify_terminal(evidence_dir):
    """Optional later read-only classification; never rewrites the old stage."""
    evidence = Path(evidence_dir)
    receipt = load(evidence / 'revocation.json')
    output = Path(receipt['pair_output'])
    stage = load(output / 'stage.json')
    jobs = stage.get('jobs', [])
    if (receipt.get('revocation_confirmed') is not True or
            Path(receipt['master_path']).exists() or
            sha(receipt['preserved_master_path']) != receipt['master_sha256'] or
            stage.get('status') != 'stopped_with_failure' or
            stage.get('error_type') != 'FileNotFoundError' or
            not any(value in stage.get('error', '') for value in
                    (receipt['master_path'], repr(receipt['master_path']))) or
            len(jobs) != 1 or jobs[0].get('label') != 'dense' or
            jobs[0].get('returncode') != 0 or jobs[0].get('status') != 'completed' or
            (output / 'sparse-protocol.json').exists() or (output / 'sparse').exists()):
        raise ValueError('Old parent did not exhibit the exact expected administrative revocation')
    if sha(output / 'dense' / 'summary.json') != jobs[0]['summary_sha256']:
        raise ValueError('Completed dense summary was altered')
    return {'status': 'verified_administrative_dispatch_revocation_after_dense_completion',
            'stage_path': str(output / 'stage.json'), 'stage_sha256': sha(output / 'stage.json'),
            'dense_summary_sha256': jobs[0]['summary_sha256'], 'sparse_jobs_launched': 0,
            'scientific_pair_complete': False, 'utc': utc()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--pair-output', type=Path, required=True)
    parser.add_argument('--controller-pid', type=int, required=True)
    parser.add_argument('--dense-pid', type=int, required=True)
    parser.add_argument('--evidence-dir', type=Path)
    parser.add_argument('--revoke', action='store_true')
    args = parser.parse_args()
    shared = (args.project_root, args.protocol, args.protocol_sha256, args.pair_output,
              args.controller_pid, args.dense_pid)
    if args.revoke:
        if args.evidence_dir is None:
            parser.error('--revoke requires a fresh --evidence-dir')
        result = revoke_dispatch(*shared, args.evidence_dir)
    else:
        result = inspect_handoff(*shared)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
