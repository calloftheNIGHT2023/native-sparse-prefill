"""Execution-only MIG adapter for the frozen one-epoch scientific worker.

No training/math changes. Default is source/data validation only. GPU access is
checked in a short-lived child before training so that its CUDA context exits.
Physical nvidia-smi permission denial is recorded, never reported as an empty
host queue. The fallback establishes container-visible isolation and free memory,
not visibility into other containers or future uncooperative processes.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_babylm_one_epoch_v0 as original

require, load, resolve, sha, write = (original.require, original.load, original.resolve,
                                    original.sha, original.write)
MIN_MEMORY_GIB, MAX_MEMORY_GIB, MIN_FREE_FRACTION = 22, 26, 0.90
GPU_DEVICE = re.compile(r"^/dev/nvidia[^/]*(?:/[^/]+)?$")
# Fixed to the observed template worker on this assigned Pod. A restarted or
# differently parented service needs a new explicit execution review.
TEMPLATE_NGINX_CHILD_PID, TEMPLATE_NGINX_PARENT_PID = 527, 526


def canonical_uuid(value):
    if isinstance(value, bytes):
        require(len(value) == 16, 'CUDA UUID bytes must contain 16 bytes')
        return str(uuid.UUID(bytes=value))
    text = str(value).strip()
    for prefix in ('MIG-', 'GPU-'):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return str(uuid.UUID(text))


def validate_hardware_config(protocol):
    hardware = protocol.get('execution_hardware', {})
    require(isinstance(hardware, dict), 'Missing execution_hardware')
    require(str(hardware.get('mig_uuid', '')).startswith('MIG-'), 'Explicit MIG UUID required')
    require(str(hardware.get('physical_uuid', '')).startswith('GPU-'), 'Explicit physical GPU UUID required')
    canonical_uuid(hardware['mig_uuid'])
    canonical_uuid(hardware['physical_uuid'])
    require(isinstance(hardware.get('driver_version'), str) and bool(hardware['driver_version']),
            'Explicit driver_version required')
    require(hardware.get('mig_profile', '1g.24gb') == '1g.24gb', 'Only the frozen 1g.24gb profile is supported')
    return hardware


def classify_compute_queue(returncode, stdout, stderr):
    """Never expose process names; numeric PIDs always block a launch."""
    rows = list(csv.reader(io.StringIO(stdout.strip()))) if stdout.strip() else []
    numeric_pids = [int(row[0].strip()) for row in rows if row and row[0].strip().isdigit()]
    require(not numeric_pids, 'nvidia-smi reports active compute PIDs: ' + ','.join(map(str, numeric_pids)))
    if returncode == 0 and not rows and not stderr.strip():
        return {'nvidia_smi_queue': 'empty_visible_queue', 'permission_degraded': False}
    denied = 'insufficient permissions' in (stdout + '\n' + stderr).lower()
    if denied:
        # Unknown process identifiers are acceptable only if every output row
        # is an explicit permissions marker, never an unrecognized queue row.
        require(all(row and all(cell.strip().lower() in ('[insufficient permissions]',
                    'insufficient permissions', '') for cell in row) for row in rows),
                'Mixed or unrecognized compute queue output; isolation is unproven')
        return {'nvidia_smi_queue': 'unavailable_insufficient_permissions',
                'permission_degraded': True, 'queue_visibility_claim': 'No empty host queue claim'}
    raise ValueError('nvidia-smi compute queue could not be verified; refusing launch')


def validate_nvidia_identity(physical_csv, listing, hardware):
    rows = list(csv.reader(io.StringIO(physical_csv.strip())))
    require(len(rows) == 1 and len(rows[0]) == 2, 'Exactly one physical GPU identity is required')
    require(rows[0][0].strip() == hardware['physical_uuid'] and
            rows[0][1].strip() == hardware['driver_version'], 'Physical GPU UUID or driver mismatch')
    physical = re.findall(r'\(UUID: (GPU-[0-9a-fA-F-]+)\)', listing)
    instances = re.findall(r'MIG\s+(\S+)\s+Device\s+\d+:\s*\(UUID:\s*(MIG-[0-9a-fA-F-]+)\)', listing)
    require(physical == [hardware['physical_uuid']], 'nvidia-smi -L physical GPU differs')
    require(instances == [('1g.24gb', hardware['mig_uuid'])], 'Require exactly the assigned 1g.24gb MIG instance')
    return {'physical_uuid': hardware['physical_uuid'], 'mig_uuid': hardware['mig_uuid'],
            'driver_version': hardware['driver_version'], 'mig_profile': '1g.24gb'}


def validate_cuda_probe(probe, hardware):
    require(probe.get('visible_cuda_device_count') == 1, 'Exactly one visible CUDA device required')
    require(probe.get('driver_visible_cuda_device_count') == 1, 'Exactly one CUDA driver device required')
    # This pinned CUDA/PyTorch runtime exposes the physical UUID in properties.
    # CUDA driver v2 UUID, unlike the legacy API, identifies the MIG instance.
    require(canonical_uuid(probe.get('torch_properties_uuid')) == canonical_uuid(hardware['physical_uuid']),
            'PyTorch physical UUID differs from the assigned physical GPU')
    require(canonical_uuid(probe.get('driver_v2_mig_uuid')) == canonical_uuid(hardware['mig_uuid']),
            'CUDA driver v2 UUID is not the assigned MIG instance')
    for key in ('properties_total_bytes', 'free_bytes', 'total_bytes'):
        require(type(probe.get(key)) is int and probe[key] > 0, 'Invalid CUDA memory measurement: ' + key)
    total, free = probe['total_bytes'], probe['free_bytes']
    require(MIN_MEMORY_GIB * 2**30 <= total <= MAX_MEMORY_GIB * 2**30,
            'CUDA memory is outside the frozen 22–26 GiB MIG range')
    require(abs(total - probe['properties_total_bytes']) <= 16 * 2**20,
            'CUDA memory total disagrees with device properties')
    require(free <= total and free / total >= MIN_FREE_FRACTION,
            'MIG has less than 90% free memory; refusing shared or occupied device')
    return {**probe, 'free_fraction': free / total, 'passed': True}


def _process_service_identity(process):
    """Read only service name, numeric UID fields and parent PID."""
    comm = (process / 'comm').read_text(encoding='utf-8').strip()
    status = {}
    for line in (process / 'status').read_text(encoding='utf-8').splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            if key in ('Name', 'Uid', 'PPid'):
                status[key] = value.strip()
    return {'pid': int(process.name), 'comm': comm, 'status_name': status.get('Name'),
            'uids': [int(value) for value in status.get('Uid', '').split()],
            'parent_pid': int(status['PPid'])}


def _template_nginx_fd_exception(process, proc_root):
    """Narrow identified service exception; unreadable child FDs stay unknown."""
    require(int(process.name) == TEMPLATE_NGINX_CHILD_PID,
            'Not the previously identified template nginx worker PID')
    child = _process_service_identity(process)
    require(child['comm'] == child['status_name'] == 'nginx'
            and child['uids'] == [65534] * 4
            and child['parent_pid'] == TEMPLATE_NGINX_PARENT_PID,
            'Template nginx child identity or UIDs changed')
    parent_path = Path(proc_root) / str(child['parent_pid'])
    parent = _process_service_identity(parent_path)
    require(parent['comm'] == parent['status_name'] == 'nginx'
            and parent['uids'] == [0] * 4 and parent['parent_pid'] == 1,
            'Template nginx parent must be the root init-owned nginx master')
    require(os.readlink(parent_path / 'exe') == '/usr/sbin/nginx',
            'Template nginx parent executable changed')
    parent_fds = list((parent_path / 'fd').iterdir())
    parent_gpu_fds = []
    for descriptor in parent_fds:
        # Parent unreadability or disappearance is never eligible for its
        # child's exception: every listed parent descriptor must be inspected.
        target = os.readlink(descriptor)
        if GPU_DEVICE.fullmatch(target.removesuffix(' (deleted)')):
            parent_gpu_fds.append(target)
    require(not parent_gpu_fds, 'Template nginx parent holds GPU device descriptors')
    require(child == _process_service_identity(process)
            and parent == _process_service_identity(parent_path),
            'Template nginx process identity changed during validation')
    return {'kind': 'explicit_template_nginx_child_fd_permission_exception',
            'pid': child['pid'], 'parent_pid': parent['pid'], 'uids': child['uids'],
            'comm': child['comm'], 'parent_uids': parent['uids'],
            'parent_executable': '/usr/sbin/nginx', 'parent_parent_pid': 1,
            'parent_descriptors_checked': len(parent_fds), 'parent_gpu_device_fds': [],
            'child_fd_visibility': 'partial_or_unavailable_permission_denied_not_fully_inspected',
            'limitation': 'Identified template service exception; child GPU FD absence is not proven. '
                          'MIG identity, single device, memory availability and lock gates remain mandatory.'}


def scan_gpu_fds(proc_root=Path('/proc'), exclude_pids=()):
    """Inspect FD targets plus the one pinned service identity, never cmdline/env."""
    excluded = set(exclude_pids)
    processes = list(Path(proc_root).iterdir())
    checked = 0
    users = []
    exceptions = {}
    for process in processes:
        if not process.name.isdigit() or int(process.name) in excluded:
            continue
        pid = int(process.name)
        try:
            descriptors = list((process / 'fd').iterdir())
            checked += 1
        except (FileNotFoundError, ProcessLookupError):
            continue  # Process terminated while enumerating /proc.
        except PermissionError as error:
            try:
                exceptions[pid] = _template_nginx_fd_exception(process, proc_root)
            except (OSError, ValueError, KeyError) as rejection:
                raise ValueError(f'Cannot inspect container PID {pid} file descriptors; '
                                 f'no eligible template-service exception: {rejection}') from error
            continue
        devices = set()
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError as error:
                try:
                    if pid not in exceptions:
                        exceptions[pid] = _template_nginx_fd_exception(process, proc_root)
                except (OSError, ValueError, KeyError) as rejection:
                    raise ValueError(f'Cannot inspect container PID {pid} descriptor; '
                                     f'no eligible template-service exception: {rejection}') from error
                continue  # Still inspect all readable child FDs; any GPU FD blocks.
            if GPU_DEVICE.fullmatch(target.removesuffix(' (deleted)')):
                devices.add(target)
        if devices:
            users.append({'pid': pid, 'device_fds': sorted(devices)})
    require(not users, 'Other container GPU device users: ' + json.dumps(users, sort_keys=True))
    return {'checked_container_processes': checked, 'other_gpu_device_users': users,
            'service_fd_permission_exceptions': list(exceptions.values()),
            'all_nonexcluded_process_fds_fully_inspected': not exceptions,
            'scope': 'current proc namespace; identified template nginx FD exception is explicit; '
                     'no cmdline or environment read', 'passed': True}


def cuda_driver_identity():
    """Read the MIG-aware CUDA driver UUID; legacy device UUID is insufficient."""
    class CUuuid(ctypes.Structure):
        _fields_ = [('bytes', ctypes.c_ubyte * 16)]
    driver = ctypes.CDLL('libcuda.so.1')
    signatures = {'cuInit': [ctypes.c_uint], 'cuDeviceGetCount': [ctypes.POINTER(ctypes.c_int)],
                  'cuDeviceGet': [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
                  'cuDeviceGetUuid_v2': [ctypes.POINTER(CUuuid), ctypes.c_int]}
    for name, args in signatures.items():
        function = getattr(driver, name)
        function.argtypes, function.restype = args, ctypes.c_int
    require(driver.cuInit(0) == 0, 'CUDA driver initialization failed')
    count, device, identity = ctypes.c_int(), ctypes.c_int(), CUuuid()
    require(driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value == 1,
            'CUDA driver must expose exactly one device')
    require(driver.cuDeviceGet(ctypes.byref(device), 0) == 0, 'Cannot resolve CUDA driver device zero')
    require(driver.cuDeviceGetUuid_v2(ctypes.byref(identity), device) == 0,
            'MIG-aware cuDeviceGetUuid_v2 failed')
    return {'driver_visible_cuda_device_count': count.value,
            'driver_v2_mig_uuid': canonical_uuid(bytes(identity.bytes))}


def cuda_probe_child(hardware):
    import torch
    count = torch.cuda.device_count()
    require(count == 1, 'Exactly one visible CUDA device required')
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    probe = {'visible_cuda_device_count': count, 'torch_properties_uuid': canonical_uuid(props.uuid),
             **cuda_driver_identity(),
             'device_name': props.name, 'properties_total_bytes': props.total_memory,
             'free_bytes': free, 'total_bytes': total, 'probe_pid': os.getpid(),
             'model_forward_calls': 0, 'optimizer_updates': 0}
    probe = validate_cuda_probe(probe, hardware)
    probe['container_fd_scan'] = scan_gpu_fds(exclude_pids={os.getpid()})
    return probe


def isolation_gate(protocol_path, hardware, audit=None):
    if audit is None:
        audit = {}
    def smi(*args):
        return subprocess.run(['nvidia-smi', *args], capture_output=True, text=True, timeout=20)
    identities = smi('--query-gpu=uuid,driver_version', '--format=csv,noheader')
    listing = smi('-L')
    require(identities.returncode == listing.returncode == 0, 'Cannot verify physical GPU/MIG identity')
    identity = validate_nvidia_identity(identities.stdout, listing.stdout, hardware)
    audit['identity'] = identity
    queue = smi('--query-compute-apps=pid,process_name', '--format=csv,noheader')
    queue_record = classify_compute_queue(queue.returncode, queue.stdout, queue.stderr)
    audit.update(queue_record)
    probe = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--cuda-isolation-probe',
                            '--protocol', str(protocol_path)], capture_output=True, text=True,
                           cwd=ROOT, timeout=60)
    require(probe.returncode == 0, 'CUDA isolation child failed: ' + probe.stderr[-2000:])
    measured = validate_cuda_probe(json.loads(probe.stdout), hardware)
    require(measured.get('container_fd_scan', {}).get('passed') is True, 'Child FD scan did not pass')
    # The child has exited; this second scan verifies no probe context remains.
    after = scan_gpu_fds()
    audit.update(cuda_probe=measured, probe_exited_before_training=True, post_probe_container_fd_scan=after,
                 isolation_scope='Pinned MIG identity, current container FD visibility and free memory; '
                                 'not proof of host-wide queue visibility or future exclusive use')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--mode', choices=['dense', 'sparse'])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--numerics', type=Path)
    parser.add_argument('--numerics-sha256')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cuda-isolation-probe', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    pp = resolve(args.protocol)
    p = load(pp)
    hardware = validate_hardware_config(p)
    if args.cuda_isolation_probe:
        require(not args.execute, 'Isolation probe cannot execute training')
        print(json.dumps(cuda_probe_child(hardware), allow_nan=False))
        return 0
    require(args.mode is not None and args.output_dir is not None, 'mode and output-dir are required')
    plan = original.validate(p)
    if not args.execute:
        print(json.dumps({**plan, 'execution_hardware': hardware}, indent=2))
        return 0
    require(os.name == 'posix', 'MIG worker requires Linux')
    require(args.protocol_sha256 and sha(pp) == args.protocol_sha256, 'Explicit frozen protocol SHA required')
    require(args.numerics is not None and args.numerics_sha256, 'Original-threshold numerical receipt required')
    require(p['source_sha256'].get('scripts/run_babylm_one_epoch_mig_worker_v0.py') == sha(Path(__file__)),
            'MIG execution adapter must be source-pinned')
    gpu, _ = original.numerical_gate(resolve(args.numerics), args.numerics_sha256)
    require(gpu == hardware['physical_uuid'], 'Numerical receipt is for a different physical GPU')
    import fcntl
    lock_path = '/tmp/babylm-one-epoch-' + hardware['mig_uuid'] + '.lock'
    with open(lock_path, 'a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = resolve(args.output_dir)
        require(out.is_relative_to(ROOT / 'results') and not out.exists(), 'Use a fresh results directory')
        out.mkdir(parents=True, exist_ok=False)
        stage = {'status': 'checking_mig_isolation', 'mode': args.mode, 'pid': os.getpid(),
                 'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(pp),
                 'plan': plan, 'execution_hardware': hardware, 'mig_lock_path': lock_path,
                 'numerics_sha256': args.numerics_sha256, 'new_numerics_calls': 0,
                 'no_automatic_resume_or_followup': True}
        write(out / 'worker-stage.json', stage)
        started = time.monotonic()
        try:
            stage['isolation_audit'] = {}
            isolation_gate(pp, hardware, stage['isolation_audit'])
            stage['status'] = 'isolation_verified'
            write(out / 'worker-stage.json', stage)
            require(sha(pp) == args.protocol_sha256, 'Protocol changed during execution checks')
            command = [sys.executable, '-u', str(ROOT / 'scripts/run_babylm_de_v0.py'),
                       '--protocol', str(pp), '--mode', args.mode, '--output-dir', str(out / 'run')]
            with (out / 'stdout.log').open('xb') as stdout, (out / 'stderr.log').open('xb') as stderr:
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
                    raise RuntimeError('Hard timeout: preserve evidence; no automatic resume')
            require(code == 0, 'Training failed; inspect preserved logs and failure checkpoint')
            subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_babylm_one_epoch_v0.py'),
                            '--run-dir', str(out / 'run'), '--output', str(out / 'tail100-audit.json')],
                           check=True, cwd=ROOT, timeout=120)
            stage.update(status='complete_one_epoch_audited', child_returncode=code)
            return 0
        except BaseException as error:
            stage.update(status='failed_or_incomplete', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            stage.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                         elapsed_wall_seconds=time.monotonic() - started)
            stage['estimated_stage_cost_usd'] = stage['elapsed_wall_seconds'] / 3600 * p['hourly_rate_usd']
            stage['cost_basis'] = 'Frozen worker wall estimate; excludes pre-launch idle; not invoice'
            write(out / 'worker-stage.json', stage)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == '__main__':
    raise SystemExit(main())
