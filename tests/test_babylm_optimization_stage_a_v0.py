import copy
import unittest

from scripts.run_babylm_optimization_stage_a_v0 import replay_audit, validate_jobs


class StageATests(unittest.TestCase):
    def fixture(self):
        jobs = [{'name': f'{role}_{mode}', 'role': role, 'replay_reference': 'frozen.json'}
                for role in ('replay', 'full_dev') for mode in ('dense', 'sparse')]
        protocols = [{'mode': mode, 'dtype': 'float32', 'device': 'cuda',
                      'checkpoint_sha256': mode, 'window_indices': list(range(48)) if role == 'replay' else None}
                     for role in ('replay', 'full_dev') for mode in ('dense', 'sparse')]
        return jobs, protocols

    def test_valid_order(self):
        validate_jobs(*self.fixture())

    def test_cannot_omit_or_reorder_replays(self):
        jobs, protocols = self.fixture()
        jobs[0]['role'] = 'full_dev'
        with self.assertRaises(ValueError):
            validate_jobs(jobs, protocols)

    def test_changed_checkpoint_rejected(self):
        jobs, protocols = self.fixture()
        protocols[-1]['checkpoint_sha256'] = 'different'
        with self.assertRaises(ValueError):
            validate_jobs(jobs, protocols)

    def test_incomplete_full_dev_rejected(self):
        jobs, protocols = self.fixture()
        protocols[-1]['window_indices'] = [0, 1]
        with self.assertRaises(ValueError):
            validate_jobs(jobs, protocols)

    def test_replay_counts_values_and_order(self):
        row = {'window_index': 8, 'loss_tokens': 2, 'nll_sum': 7.0}
        totals = {'windows': 1, 'forward_calls': 1, 'input_tokens': 3, 'loss_tokens': 2,
                  'word_exposures': 2, 'nll': 3.5, 'attention_counts': {'kept': 5}}
        reference = {'total': totals, 'position_diagnostics': {'per_window': [row]}}
        summary = {'total': copy.deepcopy(totals)}
        tol = {'aggregate_nll_abs': 1e-6, 'window_nll_abs': 1e-5}
        self.assertTrue(replay_audit(reference, summary, [row], tol)['passed'])
        for key, bad in [('loss_tokens', 3), ('nll', 3.5001), ('nll', float('nan')),
                         ('attention_counts', {'kept': 4})]:
            other = copy.deepcopy(summary)
            other['total'][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                replay_audit(reference, other, [row], tol)
        with self.assertRaises(ValueError):
            replay_audit(reference, summary, [{**row, 'window_index': 9}], tol)


if __name__ == '__main__':
    unittest.main()
