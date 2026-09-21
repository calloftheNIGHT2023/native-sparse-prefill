"""Execution-only admission for the newly assigned 48GB MIG; no model calls."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_babylm_one_epoch_mig_worker_v0 as old


def check_probe(probe, hardware):
    assert probe['visible_cuda_device_count'] == probe['driver_visible_cuda_device_count'] == 1
    assert old.canonical_uuid(probe['torch_properties_uuid']) == old.canonical_uuid(hardware['physical_uuid'])
    assert old.canonical_uuid(probe['driver_v2_mig_uuid']) == old.canonical_uuid(hardware['mig_uuid'])
    assert 44 * 2**30 <= probe['total_bytes'] <= 50 * 2**30
    assert abs(probe['total_bytes'] - probe['properties_total_bytes']) <= 16 * 2**20
    assert 0.9 <= probe['free_bytes'] / probe['total_bytes'] <= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--probe', action='store_true')
    args = ap.parse_args()
    p = json.loads(args.protocol.read_text())
    hw = p['execution_hardware']
    assert hw['mig_profile'] == '2g.48gb'
    if args.probe:
        import torch
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        result = dict(visible_cuda_device_count=torch.cuda.device_count(),
                      torch_properties_uuid=old.canonical_uuid(props.uuid),
                      properties_total_bytes=props.total_memory, total_bytes=total,
                      free_bytes=free, **old.cuda_driver_identity())
        check_probe(result, hw)
        print(json.dumps(result))
        return
    assert args.output is not None and not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(status='running', observed_utc=datetime.now(timezone.utc).isoformat(),
                   model_forward_calls=0, backward_calls=0, optimizer_updates=0,
                   execution_hardware=hw)
    try:
        # Only the Pod identifier is extracted, never full environment values.
        pod = next(x.split(b'=', 1)[1].decode() for x in Path('/proc/1/environ').read_bytes().split(b'\0')
                   if x.startswith(b'RUNPOD_POD_ID='))
        assert pod == p['pod_id']
        old.TEMPLATE_NGINX_CHILD_PID = p['service_exception']['child_pid']
        old.TEMPLATE_NGINX_PARENT_PID = p['service_exception']['parent_pid']
        lock = open('/tmp/babylm-one-epoch-' + hw['mig_uuid'] + '.lock', 'a')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        listing = subprocess.check_output(['nvidia-smi', '-L'], text=True)
        csv = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'], text=True)
        assert csv.strip() == hw['physical_uuid'] + ', ' + hw['driver_version']
        assert old.re.findall(r'\(UUID: (GPU-[0-9a-fA-F-]+)\)', listing) == [hw['physical_uuid']]
        assert old.re.findall(r'MIG\s+(\S+)\s+Device\s+\d+:\s*\(UUID:\s*(MIG-[0-9a-fA-F-]+)\)', listing) == [(hw['mig_profile'], hw['mig_uuid'])]
        queue = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True)
        receipt['queue'] = old.classify_compute_queue(queue.returncode, queue.stdout, queue.stderr)
        receipt['fd_scan_before'] = old.scan_gpu_fds()
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--protocol', str(args.protocol.resolve()), '--probe'], capture_output=True, text=True, timeout=60, check=True)
        receipt['cuda_probe'] = json.loads(child.stdout)
        check_probe(receipt['cuda_probe'], hw)
        receipt['fd_scan_after'] = old.scan_gpu_fds()
        receipt.update(status='passed', passed=True, pod_id=pod,
                       scope='Assigned MIG identity, free memory and container-visible FDs; not host-wide exclusivity')
    except BaseException as exc:
        receipt.update(status='failed', passed=False, failure_type=type(exc).__name__, failure_message=str(exc))
    finally:
        receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    if not receipt.get('passed'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
