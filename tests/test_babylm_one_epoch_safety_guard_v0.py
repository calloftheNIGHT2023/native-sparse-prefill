"""Synthetic /proc and pidfd guard checks; zero processes signalled or launched."""
import signal
import unittest
from unittest import mock
from pathlib import Path
from enum import IntEnum

from scripts import babylm_one_epoch_safety_guard_v0 as guard


class LinuxOnlySignal(IntEnum):
    SIGKILL = 9
    SIGCONT = 18
    SIGSTOP = 19


def identity(pid=6660, ticks=96380849):
    return {'pid': pid, 'ppid': 6520, 'pgid': pid, 'session': pid,
            'start_ticks': ticks, 'cwd': str(guard.REMOTE_ROOT), 'state': 'S'}


class GuardTests(unittest.TestCase):
    def test_proc_stat_with_spaces_and_parentheses(self):
        # Fields 3..22, including a command name containing parentheses.
        tail = ['S', '6520', '6660', '6660'] + ['0'] * 15 + ['96380849']
        result = guard.parse_stat('6660 (python (training)) ' + ' '.join(tail))
        self.assertEqual(result['start_ticks'], 96380849)
        self.assertEqual((result['ppid'], result['pgid'], result['session']), (6520, 6660, 6660))

    def test_reused_pid_or_cwd_change_blocks_signal(self):
        for changed in ({'start_ticks': 96380850}, {'cwd': '/other'}, {'pgid': 99}):
            with self.subTest(changed=changed), mock.patch.object(guard.os, 'pidfd_open', return_value=42, create=True), \
                    mock.patch.object(guard.os, 'close') as close, \
                    mock.patch.object(guard, 'process_identity', return_value={**identity(), **changed}), \
                    mock.patch.object(guard.signal, 'pidfd_send_signal', create=True) as send:
                with self.assertRaises(guard.Attention):
                    guard.signal_identity(identity(), signal.SIGTERM)
                send.assert_not_called()
                close.assert_called_once_with(42)

    def test_valid_pidfd_signal_and_closed_descriptor(self):
        with mock.patch.object(guard.os, 'pidfd_open', return_value=42, create=True), \
                mock.patch.object(guard.os, 'close') as close, \
                mock.patch.object(guard, 'process_identity', return_value=identity()), \
                mock.patch.object(guard.signal, 'pidfd_send_signal', create=True) as send:
            self.assertTrue(guard.signal_identity(identity(), signal.SIGTERM))
            send.assert_called_once_with(42, signal.SIGTERM, None, 0)
            close.assert_called_once_with(42)

    def test_exited_process_is_not_signalled(self):
        with mock.patch.object(guard.os, 'pidfd_open', side_effect=ProcessLookupError, create=True), \
                mock.patch.object(guard.signal, 'pidfd_send_signal', create=True) as send:
            self.assertFalse(guard.signal_identity(identity(), signal.SIGTERM))
            send.assert_not_called()

    def test_parent_loss_is_detected_even_if_train_still_alive(self):
        live = {'outer': {'pid': 6455, 'ppid': 1}, 'sequence': {'pid': 6456, 'ppid': 6455},
                'dense_worker': {'pid': 6520, 'ppid': 6456}, 'dense_train': identity()}
        self.assertIsNone(guard.orphan_reason(live))
        del live['dense_worker']
        self.assertEqual(guard.orphan_reason(live), 'dense_train_lost_registered_parent')

    def test_unowned_training_group_is_rejected(self):
        with mock.patch.object(guard, 'process_identity', return_value=None), self.assertRaises(guard.Attention):
            guard.owned_group({**identity(), 'pgid': 6455})

    def test_bounded_shutdown_freezes_dispatch_then_kills_stubborn_training(self):
        registry = {'outer': {**identity(6455, 1), 'ppid': 1},
                    'sequence': {**identity(6456, 2), 'ppid': 6455, 'pgid': 6455},
                    'dense_worker': {**identity(6520, 3), 'ppid': 6456, 'pgid': 6455},
                    'dense_train': identity()}
        alive = {row['pid'] for row in registry.values()}
        sent = []
        clock = iter(range(0, 10000, 100))
        def fake_signal(row, sig):
            sent.append((row['pid'], sig))
            if row['pid'] not in alive:
                return False
            if sig == LinuxOnlySignal.SIGKILL or (sig == signal.SIGTERM and row['pid'] != 6660):
                alive.remove(row['pid'])
            return True
        with mock.patch.object(signal, 'SIGSTOP', LinuxOnlySignal.SIGSTOP, create=True), \
                mock.patch.object(signal, 'SIGCONT', LinuxOnlySignal.SIGCONT, create=True), \
                mock.patch.object(signal, 'SIGKILL', LinuxOnlySignal.SIGKILL, create=True), \
                mock.patch.object(guard, 'signal_identity', side_effect=fake_signal), \
                mock.patch.object(guard, 'collect', return_value={}), \
                mock.patch.object(guard, 'read_json', return_value={'active_mode': 'dense'}), \
                mock.patch.object(Path, 'iterdir', return_value=iter([])), \
                mock.patch.object(guard, 'owned_group', side_effect=lambda row: [row] if row['pid'] in alive else []), \
                mock.patch.object(guard, 'live_registered', side_effect=lambda rows: {k: v for k, v in rows.items() if v['pid'] in alive}), \
                mock.patch.object(guard.time, 'monotonic', side_effect=lambda: next(clock)), \
                mock.patch.object(guard.time, 'sleep'):
            audit = {}
            guard.stop_owned(registry, audit)
        self.assertTrue(audit['owned_processes_verified_stopped'])
        self.assertFalse(alive)
        self.assertEqual(sent[0], (6456, LinuxOnlySignal.SIGSTOP))
        self.assertLess(sent.index((6660, signal.SIGTERM)), sent.index((6660, LinuxOnlySignal.SIGKILL)))
        self.assertLess(sent.index((6660, LinuxOnlySignal.SIGKILL)), sent.index((6520, signal.SIGTERM)))


if __name__ == '__main__':
    unittest.main()
