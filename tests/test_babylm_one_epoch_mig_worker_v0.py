"""GPU-free execution-guard tests; no CUDA/model/cloud calls."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import unittest
from unittest import mock

from scripts import run_babylm_one_epoch_mig_worker_v0 as worker


HARDWARE = {'physical_uuid': 'GPU-753a81b8-728e-2d13-ea85-0f1786e6bd74',
            'mig_uuid': 'MIG-e8b41c86-543f-5196-81bd-05a67c68eaf3',
            'driver_version': '580.01', 'mig_profile': '1g.24gb'}


@contextmanager
def nginx_proc_fixture(child_name='nginx', child_uid=65534, child_parent=526,
                       parent_name='nginx', parent_uid=0, parent_parent=1,
                       parent_exe='/usr/sbin/nginx', parent_gpu=False,
                       child_readable_gpu=False, parent_fd_denied=False):
    def children(path):
        return iter({'/mock-proc': [Path('/mock-proc/526'), Path('/mock-proc/527')],
                     '/mock-proc/526/fd': [Path('/mock-proc/526/fd/3')],
                     '/mock-proc/527/fd': [Path('/mock-proc/527/fd/0'), Path('/mock-proc/527/fd/1')]}[path.as_posix()])
    def read_text(path, **kwargs):
        pid, field = path.parent.name, path.name
        name, uid, parent = ((child_name, child_uid, child_parent) if pid == '527'
                             else (parent_name, parent_uid, parent_parent))
        if field == 'comm':
            return name + '\n'
        if field == 'status':
            return f'Name:\t{name}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nPPid:\t{parent}\n'
        raise AssertionError('Unexpected process file read: ' + path.as_posix())
    def readlink(path):
        name = Path(path).as_posix()
        if name == '/mock-proc/526/exe':
            return parent_exe
        if name == '/mock-proc/527/fd/0' or (parent_fd_denied and name == '/mock-proc/526/fd/3'):
            raise PermissionError('template worker is not ptrace-readable')
        if ((parent_gpu and name == '/mock-proc/526/fd/3')
                or (child_readable_gpu and name == '/mock-proc/527/fd/1')):
            return '/dev/nvidia0'
        return '/dev/null'
    with mock.patch.object(Path, 'iterdir', children), mock.patch.object(Path, 'read_text', read_text), \
            mock.patch.object(worker.os, 'readlink', readlink):
        yield


class MigGuardTests(unittest.TestCase):
    def test_permission_marker_is_degraded_not_empty(self):
        result = worker.classify_compute_queue(0, '[Insufficient Permissions], [Insufficient Permissions]', '')
        self.assertTrue(result['permission_degraded'])
        self.assertEqual(result['nvidia_smi_queue'], 'unavailable_insufficient_permissions')

    def test_actual_or_mixed_process_rows_are_rejected(self):
        for text in ('123, python', '[Insufficient Permissions], [Insufficient Permissions]\n123, python',
                     'N/A, [Insufficient Permissions]'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                worker.classify_compute_queue(0, text, '')

    def test_empty_and_unknown_failures(self):
        self.assertFalse(worker.classify_compute_queue(0, '', '')['permission_degraded'])
        with self.assertRaises(ValueError):
            worker.classify_compute_queue(1, '', 'unknown driver failure')

    def test_exact_identity_and_multiple_mig(self):
        physical = HARDWARE['physical_uuid'] + ', ' + HARDWARE['driver_version']
        listing = f"GPU 0: RTX PRO 6000 (UUID: {HARDWARE['physical_uuid']})\n  MIG 1g.24gb Device 0: (UUID: {HARDWARE['mig_uuid']})"
        self.assertEqual(worker.validate_nvidia_identity(physical, listing, HARDWARE)['mig_uuid'], HARDWARE['mig_uuid'])
        with self.assertRaises(ValueError):
            worker.validate_nvidia_identity(physical, listing + '\n' + listing.splitlines()[1], HARDWARE)
        with self.assertRaises(ValueError):
            worker.validate_nvidia_identity(physical.replace('580.01', '580.02'), listing, HARDWARE)

    def test_cuda_probe_identity_capacity_and_occupancy(self):
        probe = {'visible_cuda_device_count': 1, 'driver_visible_cuda_device_count': 1,
                 'torch_properties_uuid': HARDWARE['physical_uuid'], 'driver_v2_mig_uuid': HARDWARE['mig_uuid'],
                 'properties_total_bytes': 24 * 2**30, 'total_bytes': 24 * 2**30,
                 'free_bytes': 23 * 2**30}
        self.assertTrue(worker.validate_cuda_probe(probe, HARDWARE)['passed'])
        for patch in ({'driver_v2_mig_uuid': HARDWARE['physical_uuid']}, {'visible_cuda_device_count': 2},
                      {'torch_properties_uuid': HARDWARE['mig_uuid']}, {'driver_visible_cuda_device_count': 2},
                      {'free_bytes': 20 * 2**30}, {'total_bytes': 48 * 2**30},
                      {'properties_total_bytes': 23 * 2**30}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                worker.validate_cuda_probe({**probe, **patch}, HARDWARE)

    def test_fd_scan_blocks_gpu_users_without_reading_command_lines(self):
        def children(path):
            return iter({'/mock-proc': [Path('/mock-proc/1'), Path('/mock-proc/2')],
                         '/mock-proc/1/fd': [Path('/mock-proc/1/fd/3')],
                         '/mock-proc/2/fd': [Path('/mock-proc/2/fd/4')]}[path.as_posix()])
        with mock.patch.object(Path, 'iterdir', children), mock.patch.object(worker.os, 'readlink', return_value='/dev/nvidia-caps/nvidia-cap3'):
            with self.assertRaisesRegex(ValueError, 'Other container GPU device users'):
                worker.scan_gpu_fds(Path('/mock-proc'), exclude_pids={1})
        with mock.patch.object(Path, 'iterdir', children), mock.patch.object(worker.os, 'readlink', return_value='/dev/null'):
            result = worker.scan_gpu_fds(Path('/mock-proc'), exclude_pids={1})
            self.assertEqual(result['checked_container_processes'], 1)

    def test_fd_permission_failure_blocks_launch(self):
        def children(path):
            if path.as_posix() == '/mock-proc':
                return iter([Path('/mock-proc/2')])
            raise PermissionError('denied')
        with mock.patch.object(Path, 'iterdir', children), self.assertRaisesRegex(ValueError, 'Cannot inspect container PID 2'):
            worker.scan_gpu_fds(Path('/mock-proc'))

    def test_identified_nginx_exception_is_explicit_not_full_fd_visibility(self):
        with nginx_proc_fixture():
            result = worker.scan_gpu_fds(Path('/mock-proc'))
        self.assertFalse(result['all_nonexcluded_process_fds_fully_inspected'])
        self.assertEqual(len(result['service_fd_permission_exceptions']), 1)
        exception = result['service_fd_permission_exceptions'][0]
        self.assertEqual((exception['pid'], exception['parent_pid']), (527, 526))
        self.assertEqual(exception['uids'], [65534] * 4)
        self.assertEqual(exception['parent_gpu_device_fds'], [])
        self.assertIn('not_fully_inspected', exception['child_fd_visibility'])

    def test_nginx_wrong_identity_parent_or_executable_is_rejected(self):
        for patch in ({'child_name': 'python'}, {'child_uid': 1000}, {'child_parent': 99},
                      {'parent_name': 'python'}, {'parent_uid': 65534}, {'parent_parent': 2},
                      {'parent_exe': '/tmp/nginx'}):
            with self.subTest(patch=patch), nginx_proc_fixture(**patch), self.assertRaises(ValueError):
                worker.scan_gpu_fds(Path('/mock-proc'))

    def test_nginx_readable_child_or_parent_gpu_descriptor_still_rejected(self):
        for patch in ({'parent_gpu': True}, {'child_readable_gpu': True}):
            with self.subTest(patch=patch), nginx_proc_fixture(**patch), self.assertRaises(ValueError):
                worker.scan_gpu_fds(Path('/mock-proc'))

    def test_nginx_unreadable_parent_is_not_excepted(self):
        with nginx_proc_fixture(parent_fd_denied=True), self.assertRaises(ValueError):
            worker.scan_gpu_fds(Path('/mock-proc'))


if __name__ == '__main__':
    unittest.main()
