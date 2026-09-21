"""Finite execution safeguard for the already-running September 19 D/E pair.

Default is plan-only. No GPU probes, models, datasets, retries or job launches.
Linux pidfds pin signal recipients; /proc cmdline and environ are never read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path('/workspace/native-sparse-prefill/babylm-one-epoch-20260919-v1')
OUTPUT = Path('results/blackwell-mig-one-epoch-pair-20260919-v3')
LAUNCH_UTC = '2026-09-19T19:43:45.353180+00:00'
DEADLINE = datetime.fromisoformat(LAUNCH_UTC).timestamp() + 45000
POLL_SECONDS, TERM_GRACE_SECONDS = 10, 30
LOG = Path('logs/one-epoch-safety-guard-20260919-v3.json')
PINNED = {'outer': (6455, 96380561), 'sequence': (6456, 96380562),
          'dense_worker': (6520, 96380582), 'dense_train': (6660, 96380849)}
TERMINAL = {'complete_paired_audited', 'failed_or_incomplete'}


class Attention(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Attention(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def parse_stat(text):
    pid_text, _, tail = text.partition(' (')
    _, marker, suffix = tail.rpartition(') ')
    require(bool(marker), 'Malformed process stat')
    fields = suffix.split()
    require(len(fields) >= 20, 'Truncated process stat')
    return {'pid': int(pid_text), 'state': fields[0], 'ppid': int(fields[1]),
            'pgid': int(fields[2]), 'session': int(fields[3]), 'start_ticks': int(fields[19])}


def process_identity(pid, with_cwd=True):
    try:
        result = parse_stat((Path('/proc') / str(pid) / 'stat').read_text(encoding='utf-8'))
        if result['state'] == 'Z':
            return None
        if with_cwd:
            result['cwd'] = os.readlink(Path('/proc') / str(pid) / 'cwd')
        return result
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_identity(expected, observed):
    if observed is None:
        return False
    for field in ('pid', 'start_ticks', 'pgid', 'session', 'cwd'):
        require(expected[field] == observed[field], f'Process identity mismatch for PID {expected["pid"]}: {field}')
    return True


def owned_group(leader):
    current = process_identity(leader['pid'])
    if current is not None:
        same_identity(leader, current)
    require(leader['pgid'] == leader['session'] == leader['pid'], 'Training must own its process group/session')
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        identity = process_identity(int(path.name), with_cwd=False)
        if identity is None or identity['pgid'] != leader['pgid']:
            continue
        require(identity['session'] == leader['session'], 'Unexpected session in owned training group')
        identity = process_identity(identity['pid'])
        if identity is None:
            continue
        require(identity['cwd'] == str(REMOTE_ROOT), 'Training group member has unexpected cwd')
        require(identity['start_ticks'] >= leader['start_ticks'], 'Older process in training group')
        members.append(identity)
    return members


def signal_identity(expected, signum):
    """Pin the kernel process before checking /proc, then signal via that pidfd."""
    try:
        descriptor = os.pidfd_open(expected['pid'], 0)
    except ProcessLookupError:
        return False
    try:
        if not same_identity(expected, process_identity(expected['pid'])):
            return False
        try:
            signal.pidfd_send_signal(descriptor, signum, None, 0)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(descriptor)


def collect(registry):
    sequence = read_json(ROOT / OUTPUT / 'sequence-stage.json')
    require(sequence is not None and sequence.get('pid') == PINNED['sequence'][0], 'Unexpected sequence stage')
    records = {'sequence_stage_status': sequence.get('status'), 'worker_stages': {}}
    for mode in ('dense', 'sparse'):
        stage = read_json(ROOT / OUTPUT / mode / 'worker-stage.json')
        if stage is None:
            continue
        records['worker_stages'][mode] = {'status': stage.get('status'), 'pid': stage.get('pid'),
                                         'child_pid': stage.get('child_pid')}
        for suffix, key, parent_label in (('worker', 'pid', 'sequence'), ('train', 'child_pid', mode + '_worker')):
            pid = stage.get(key)
            if pid is None:
                continue
            require(type(pid) is int and pid > 1, 'Invalid staged process PID')
            label = mode + '_' + suffix
            observed = process_identity(pid)
            if label in registry:
                require(registry[label]['pid'] == pid, 'Staged process PID changed without authorization')
                if observed is not None:
                    same_identity(registry[label], observed)
                continue
            if observed is None:
                # A completed short-lived process needs no signal registration.
                continue
            require(observed['cwd'] == str(REMOTE_ROOT), 'New process cwd differs')
            require(parent_label in registry and observed['ppid'] == registry[parent_label]['pid'],
                    'Cannot establish new process ownership from the registered parent chain')
            parent = process_identity(registry[parent_label]['pid'])
            require(same_identity(registry[parent_label], parent), 'Registered parent is no longer alive')
            if suffix == 'train':
                require(observed['pgid'] == observed['session'] == pid, 'New training process has unexpected group/session')
            registry[label] = observed
    return records


def live_registered(registry):
    live = {}
    for label, identity in registry.items():
        observed = process_identity(identity['pid'])
        if observed is not None:
            same_identity(identity, observed)
            live[label] = observed
    return live


def orphan_reason(live):
    parents = {'sequence': 'outer', 'dense_worker': 'sequence', 'sparse_worker': 'sequence',
               'dense_train': 'dense_worker', 'sparse_train': 'sparse_worker'}
    for label, parent in parents.items():
        if label in live and (parent not in live or live[label]['ppid'] != live[parent]['pid']):
            return label + '_lost_registered_parent'
    return None


def stop_owned(registry, audit):
    """Freeze only known supervisors, terminate verified training, then supervisors."""
    audit['shutdown_signals'] = []
    def send(label, identity, sig):
        sent = signal_identity(identity, sig)
        audit['shutdown_signals'].append({'label': label, 'pid': identity['pid'],
                                          'start_ticks': identity['start_ticks'], 'signal': sig.name, 'sent': sent})
    # Stop dispatch before it can launch the next already-authorized mode while
    # termination examines the active group. SIGSTOP does not stop training.
    if 'sequence' in registry:
        send('sequence', registry['sequence'], signal.SIGSTOP)
    # Resolve a worker forked immediately before sequence-stage.worker_pid was
    # published. The frozen dispatcher only spawns its active mode's worker.
    sequence_stage = read_json(ROOT / OUTPUT / 'sequence-stage.json') or {}
    active_mode = sequence_stage.get('active_mode')
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        candidate = process_identity(int(path.name), with_cwd=False)
        if candidate is None or candidate['ppid'] != registry['sequence']['pid']:
            continue
        candidate = process_identity(candidate['pid'])
        if candidate is None:
            continue
        require(active_mode in ('dense', 'sparse') and candidate['cwd'] == str(REMOTE_ROOT)
                and candidate['pgid'] == registry['sequence']['pgid'],
                'Unpublished dispatcher child cannot be verified')
        label = active_mode + '_worker'
        if label in registry:
            same_identity(registry[label], candidate)
        else:
            registry[label] = candidate
    for label in ('dense_worker', 'sparse_worker'):
        if label in registry:
            send(label, registry[label], signal.SIGSTOP)
    collect(registry)
    # A supervisor could have forked its training child just before publishing
    # child_pid. Register only an exact direct child with its own session/root.
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        candidate = process_identity(int(path.name), with_cwd=False)
        if candidate is None:
            continue
        for mode in ('dense', 'sparse'):
            parent = registry.get(mode + '_worker')
            if parent and candidate['ppid'] == parent['pid'] and candidate['pgid'] == candidate['session'] == candidate['pid']:
                candidate = process_identity(candidate['pid'])
                require(candidate is not None and candidate['cwd'] == str(REMOTE_ROOT), 'Unpublished child identity cannot be verified')
                label = mode + '_train'
                if label in registry:
                    same_identity(registry[label], candidate)
                else:
                    registry[label] = candidate
    groups = {label: identity for label, identity in registry.items() if label.endswith('_train')}
    for label, leader in groups.items():
        for member in owned_group(leader):
            send(label, member, signal.SIGTERM)
    end = time.monotonic() + TERM_GRACE_SECONDS
    while time.monotonic() < end and any(owned_group(leader) for leader in groups.values()):
        time.sleep(1)
    for label, leader in groups.items():
        for member in owned_group(leader):
            send(label, member, signal.SIGKILL)
    # Supervisors were frozen to prevent a D->E transition during teardown.
    for label in ('dense_worker', 'sparse_worker', 'sequence', 'outer'):
        if label in registry:
            send(label, registry[label], signal.SIGTERM)
            send(label, registry[label], signal.SIGCONT)
    end = time.monotonic() + TERM_GRACE_SECONDS
    while time.monotonic() < end and live_registered(registry):
        time.sleep(1)
    for label, identity in registry.items():
        send(label, identity, signal.SIGKILL)
    end = time.monotonic() + 10
    while time.monotonic() < end and (live_registered(registry) or any(owned_group(x) for x in groups.values())):
        time.sleep(1)
    require(not live_registered(registry) and not any(owned_group(x) for x in groups.values()),
            'Owned processes remain after bounded shutdown')
    audit['owned_processes_verified_stopped'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = {'root': str(REMOTE_ROOT), 'output': OUTPUT.as_posix(), 'pinned_processes': PINNED,
            'launch_utc': LAUNCH_UTC, 'deadline_utc': datetime.fromtimestamp(DEADLINE, timezone.utc).isoformat(),
            'poll_seconds': POLL_SECONDS, 'model_calls': 0, 'gpu_probes': 0,
            'new_training_launches': 0, 'termination': 'Verified process identities using pidfds only'}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    require(os.name == 'posix' and ROOT == REMOTE_ROOT, 'Guard is fixed to the assigned Linux project root')
    require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'Linux pidfd support is required')
    import fcntl
    with (ROOT / LOG.with_suffix('.lock')).open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (ROOT / LOG).exists(), 'Guard evidence already exists; do not overwrite or duplicate')
        audit = {'status': 'registering', 'guard_pid': os.getpid(), 'started_utc': utc(), 'plan': plan,
                 'counts': {'guard_model_forwards': 0, 'guard_backwards': 0, 'guard_optimizer_updates': 0,
                            'guard_gpu_probes': 0, 'new_training_launches': 0}, 'registry': {}}
        registry = audit['registry']
        try:
            for label, (pid, ticks) in PINNED.items():
                identity = process_identity(pid)
                require(identity is not None and identity['start_ticks'] == ticks and identity['cwd'] == str(REMOTE_ROOT),
                        'Pinned initial process is absent or replaced: ' + label)
                registry[label] = identity
            require(os.getpgrp() != registry['outer']['pgid'], 'Safety guard must run in an independent session/group')
            while True:
                audit['observed'] = collect(registry)
                live = live_registered(registry)
                reason = 'fixed_deadline' if time.time() >= DEADLINE else orphan_reason(live)
                audit.update(status='monitoring', updated_utc=utc(), live_labels=sorted(live))
                elapsed = max(0, time.time() - datetime.fromisoformat(LAUNCH_UTC).timestamp())
                audit['pair_wall_cost_estimate'] = {'elapsed_since_launch_seconds': elapsed,
                    'compute_usd_at_0_59_per_hour': elapsed / 3600 * 0.59,
                    'conservative_usd_at_0_81_per_hour': elapsed / 3600 * 0.81,
                    'basis': 'Pair wall time including possible idle; excludes pre-launch setup; not invoice'}
                write(ROOT / LOG, audit)
                if reason:
                    audit.update(status='bounded_shutdown', shutdown_reason=reason)
                    write(ROOT / LOG, audit)
                    stop_owned(registry, audit)
                    audit['status'] = 'bounded_shutdown_complete'
                    return 0
                active_training = any(label.endswith('_train') for label in live)
                if audit['observed']['sequence_stage_status'] in TERMINAL and not active_training:
                    audit['status'] = 'finished_with_terminal_sequence_and_no_training'
                    return 0
                time.sleep(min(POLL_SECONDS, max(0.01, DEADLINE - time.time())))
        except BaseException as error:
            audit.update(status='attention_required', error_type=type(error).__name__, error=str(error),
                         no_unverified_process_signalled=True)
            return 2
        finally:
            audit['finished_or_last_update_utc'] = utc()
            write(ROOT / LOG, audit)


if __name__ == '__main__':
    raise SystemExit(main())
