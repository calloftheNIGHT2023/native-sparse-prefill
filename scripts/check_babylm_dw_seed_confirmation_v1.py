"""Model-free schema and raw-evaluation audit regression checks for D/W queue."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_babylm_dw_seed_confirmation_v1 as q


def main():
    out = ROOT / 'logs/babylm-dw-seed-duration-v1-tests-20260921'
    out.mkdir(exist_ok=False)
    checks = []
    def check(name, call, rejected=False):
        try:
            call()
        except (ValueError, KeyError, TypeError, AssertionError):
            if not rejected: raise
        else:
            if rejected: raise AssertionError('Failed to reject ' + name)
        checks.append({'name': name, 'passed': True})
    reference = q.load(ROOT / 'configs/babylm-stage-w-local-20260921-v0/reference-dense.json')
    master = {'backbone_seed': 20260921, 'source_sha256': {'placeholder': 'only_static'},
              'train_timeout_seconds': 21600, 'hardware': {'physical_uuid': 'GPU-test', 'driver_version': 'test'}}
    d = copy.deepcopy(reference)
    d.update(backbone_seed=20260921, source_sha256=master['source_sha256'], scientific_condition='D_dense',
             actual_policy='dense', engine_compatibility_mode='dense', max_wall_seconds=21500,
             hourly_rate_usd=None, actual_compute_hourly_rate_usd=None, paid_ceiling_usd=None,
             execution_hardware=master['hardware'], cost_mode='time_bounded_unknown_rate',
             condition_pair='D_W_new_initialization_confirmation', seed_scope='initialization_only_data_order_unchanged')
    w = copy.deepcopy(d)
    w.update(scientific_condition='W_fixed_local', actual_policy='local', aux_weight=0,
             local_attention_contract={'block_size': 4, 'selected_complete_blocks': 64,
               'selection': 'most_recent_complete_blocks_plus_original_causal_tail', 'indexer_present': False})
    check('D_original_science_new_seed', lambda: q.validate_training(d, reference, master, 'D'))
    check('W_original_local_new_seed', lambda: q.validate_training(w, reference, master, 'W'))
    for key, value in [('learning_rate', 0.001), ('data_order_seed', 0), ('max_epochs', 2),
                       ('backbone_seed', 20260917), ('max_wall_seconds', 21600), ('scope', 'gpu_preflight')]:
        p = copy.deepcopy(d); p[key] = value
        check('reject_training_' + key, lambda p=p: q.validate_training(p, reference, master, 'D'), True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        quotient, remainder = divmod(17418742, 18791)
        rows = []
        for i in range(18792):
            n = 0 if i == 0 else quotient + int(i <= remainder)
            rows.append({'window_index': i, 'loss_tokens': n, 'input_tokens': n + 1,
                         'nll': 2.0 if n else None, 'nll_sum': 2.0 * n,
                         'ppl': math.exp(2.0) if n else None, 'forward_calls': 1,
                         'grad_enabled': False, 'model_training': False})
        p = {'expected_runtime': {}, 'expected_source_hashes': {}, 'checkpoint_sha256': 'f' * 64}
        s = {'status': 'evaluation_complete', 'partial_metrics_only': False,
             'counts': {'forward_attempts': 18792, 'forward_calls': 18792, 'committed_windows': 18792},
             'optimizer_updates': 0, 'backward_calls': 0, 'observed_window_indices': list(range(18792)),
             'protocol_sha256': q.canonical(p), 'runtime': {}, 'source_hashes': {},
             'checkpoint': {'sha256': 'f' * 64, 'point': {'updates': 1413}},
             'total': {'loss_tokens': 17418742, 'input_tokens': 17437534, 'nll': 2., 'ppl': math.exp(2.)}}
        def run(rows_arg=rows, summary=s, arm='D'):
            (directory / 'windows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows_arg))
            q.write(directory / 'summary.json', summary)
            return q.audit_eval(directory, p, arm)
        check('complete_raw_D_with_zero_target', lambda: q.require(run()['condition'] == 'D_dense', 'D mislabelled'))
        check('complete_raw_W_with_zero_target', lambda: q.require(run(arm='W')['condition'] == 'W_fixed_local', 'W mislabelled'))
        bad = copy.deepcopy(s); bad['status'] = 'evaluation_failed'; bad['partial_metrics_only'] = True
        check('reject_partial', lambda: run(summary=bad), True)
        bad = copy.deepcopy(rows); bad[-1]['window_index'] = 0
        check('reject_duplicate_window', lambda: run(rows_arg=bad), True)
        bad = copy.deepcopy(rows); bad[0]['nll'] = 0
        check('reject_zero_target_numeric_nll', lambda: run(rows_arg=bad), True)
        bad = copy.deepcopy(rows); bad[1]['grad_enabled'] = True
        check('reject_eval_grad', lambda: run(rows_arg=bad), True)
        bad = copy.deepcopy(s); bad['counts']['forward_attempts'] += 1
        check('reject_unaccounted_attempt', lambda: run(summary=bad), True)
        bad = copy.deepcopy(s); bad['total']['nll'] += 0.001
        check('reject_false_token_average', lambda: run(summary=bad), True)
    check('standalone_worker_rejected_before_models', lambda: q.worker(Path('unused'), 'unused', 'train_D'), True)
    q.write(out / 'report.json', {'status': 'passed', 'checks': checks, 'count': len(checks),
                                 'source_sha256': q.sha(ROOT / q.SELF), 'forward_calls': 0,
                                 'backward_calls': 0, 'optimizer_updates': 0,
                                 'torch_imported': 'torch' in sys.modules})
    print(json.dumps({'status': 'passed', 'checks': len(checks), 'model_calls': 0, 'torch_imported': 'torch' in sys.modules}))


if __name__ == '__main__':
    main()
