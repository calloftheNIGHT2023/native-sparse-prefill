"""Read-only final W portability gate: original 48 windows, unchanged tolerances.

The enclosing process owns the shared GPU lock and 300--600 second hard limit.
Default CLI performs static evidence validation; --execute requires master SHA.
No optimizer/RNG restore, training, checkpoint mutation, full-dev dispatch or retry.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_babylm_local_train_v0 as common
from scripts import run_babylm_local_eval_v0 as local_eval
require, resolve, sha, load, write, canonical = common.require, common.resolve, common.sha, common.load, common.write, common.canonical
SELF = 'scripts/run_babylm_local_migration_replay_v0.py'
TOLERANCE = {'aggregate_nll_abs': 1e-6, 'window_nll_abs': 1e-5}
TOTAL_COUNT_KEYS = ('windows', 'word_exposures', 'input_tokens', 'loss_tokens', 'forward_calls', 'zero_target_windows')
ROW_KEYS = ('window_index', 'source', 'segment_index_in_source', 'source_token_start',
            'source_token_end', 'input_tokens', 'loss_tokens')


def validate_runtime(reference, candidate):
    require(isinstance(reference, dict) and set(reference) == set(candidate), 'Runtime schema differs')
    require({k: v for k, v in reference.items() if k != 'gpu'} == {k: v for k, v in candidate.items() if k != 'gpu'},
            'Framework, CUDA build, numpy, threads or determinism differs from training')
    require(candidate['torch_num_threads'] == 1 and candidate['float32_matmul_precision'] == 'highest'
            and candidate['cuda_matmul_allow_tf32'] is False and candidate['cudnn_allow_tf32'] is False
            and candidate['cudnn_deterministic'] is True and candidate['cudnn_benchmark'] is False
            and candidate['deterministic_algorithms'] is True and candidate['deterministic_warn_only'] is False,
            'Original FP32 deterministic settings are required')
    require(isinstance(reference['gpu'], dict) and isinstance(candidate['gpu'], dict), 'Both GPU identities are required')
    return {'changed_fields': ['gpu.' + k for k in sorted(set(reference['gpu']) | set(candidate['gpu']))
                               if reference['gpu'].get(k) != candidate['gpu'].get(k)],
            'software_and_precision_exact': True, 'hardware_change_requires_numerical_replay': True}


def validate(master):
    require(master['schema_version'] == 1 and master['condition'] == common.CONDITION, 'Explicit W migration schema required')
    require(master.get('tolerance', TOLERANCE) == TOLERANCE, 'Frozen replay tolerances cannot change')
    require(300 <= master['hard_timeout_seconds'] <= 600 and
            0 < master['max_wall_seconds'] <= min(540, master['hard_timeout_seconds'] - 30), 'Replay wall limits differ')
    require(master['expected_machine'] == 'aarch64', 'This gate is frozen for the ARM migration')
    validate_runtime(master['reference_runtime'], master['expected_runtime'])
    sources = master['source_sha256']
    require(SELF in sources and sha(__file__) == sources[SELF], 'Migration runner source must be pinned')
    for name, digest in sources.items():
        p = resolve(name); require(p.is_relative_to(ROOT) and sha(p) == digest, 'Migration source differs: ' + name)
    sp = resolve(master['source_eval_protocol_path'])
    require(sha(sp) == master['source_eval_protocol_sha256'], 'Bound original W full-dev protocol differs')
    full = load(sp)
    audit = local_eval.validate(full)
    require(isinstance(full.get('transformers_gdn_source_sha256'), str)
            and len(full['transformers_gdn_source_sha256']) == 64, 'Original Transformers GDN source SHA must be pinned')
    require(full['expected_source_hashes'] == audit['source_sha256'], 'Original evaluation must preserve every training source pin')
    require(sources == dict(audit['source_sha256'], **{SELF: sources[SELF]}), 'Migration source set must be original source set plus this runner only')
    pp = resolve(master['training_protocol_path'])
    require(sha(pp) == master['training_protocol_file_sha256'], 'Original W training protocol bytes differ')
    training = load(pp)
    require(canonical(training) == audit['protocol_sha256'] and training['scientific_condition'] == common.CONDITION
            and training['aux_weight'] == 0 and training['actual_policy'] == 'local', 'W training protocol identity differs')
    ep = resolve(master['training_events_path'])
    require(sha(ep) == master['training_events_sha256'] == audit['events_sha256'], 'Complete original training log differs')
    raw = ep.read_bytes(); require(raw.endswith(b'\n'), 'Original training log ends with partial JSON')
    events = [json.loads(line) for line in raw.splitlines()]
    require([e['event_id'] for e in events] == list(range(1, len(events) + 1)), 'Original event sequence differs')
    require(events[-1]['type'] == 'run_stop' and events[-1]['status'] == 'epoch_complete'
            and events[-1]['counts'] == audit['counts'], 'Original W epoch is not complete')
    matches = [e for e in events if e['type'] == 'evaluation' and e['counts']['updates'] == 1413]
    require(len(matches) == 1, 'Exactly one original final1413 panel is required')
    reference = matches[0]['metrics']; indices = training['eval_window_indices']
    require(reference['status'] == 'nll_evaluation_complete' and reference['optimizer_updates'] == 0
            and reference['grad_enabled_during_model_forward'] is False and reference['auxiliary_loss_included_in_nll'] is False,
            'Original final panel must be pure no-grad LM evaluation')
    require(reference['window_indices'] == indices and len(indices) == len(set(indices)) == 48
            and [r['window_index'] for r in reference['position_diagnostics']['per_window']] == indices,
            'Original fixed48 order differs')
    require(reference['manifest_sha256'] == full['dev_manifest_sha256'] == training['eval_manifest_sha256'], 'Fixed dev manifest differs')
    require(reference['total']['windows'] == reference['total']['forward_calls'] == 48
            and reference['total']['input_tokens'] == 77755 and reference['total']['loss_tokens'] == 77707,
            'Original fixed48 denominators differ')
    require(full['model_config'] == training['model_config'] and
            all(full[k] == training[k] for k in ('backbone_seed', 'indexer_seed')), 'Full protocol model/seeds differ')
    return {'full_eval_protocol': full, 'training_protocol': training, 'training_audit': audit,
            'reference': reference, 'reference_event_canonical_sha256': canonical(matches[0]), 'indices': indices}


def derive_evaluation(master, plan, runtime):
    child = copy.deepcopy(plan['full_eval_protocol'])
    child.update(scope='engineering', window_indices=plan['indices'], expected_runtime=runtime,
                 expected_source_hashes=master['source_sha256'], max_wall_seconds=master['max_wall_seconds'],
                 output_dir=str(resolve(master['output_dir']) / 'evaluation'),
                 migration_replay=True, quality_result=False)
    return child


def compare_replay(reference, summary, rows, cfg):
    require(summary['status'] == 'evaluation_complete' and not summary['partial_metrics_only'], 'Replay evaluator incomplete')
    require(summary['counts']['forward_attempts'] == summary['counts']['forward_calls'] == summary['counts']['committed_windows'] == 48
            and summary['backward_calls'] == summary['optimizer_updates'] == 0, 'Replay physical call counts differ')
    old_total, new_total = reference['total'], summary['total']
    for k in TOTAL_COUNT_KEYS: require(new_total[k] == old_total[k], 'Replay total count differs: ' + k)
    require(new_total['attention_counts'] == old_total['attention_counts'], 'Replay total attention support/counts differ')
    old_rows = reference['position_diagnostics']['per_window']
    require(len(rows) == len(old_rows) == 48 and [r['window_index'] for r in rows] == reference['window_indices'], 'Replay row order/coverage differs')
    require(abs(math.fsum(r['nll_sum'] for r in rows) / new_total['loss_tokens'] - new_total['nll']) <= 1e-6,
            'Replay per-window reduction disagrees with aggregate NLL')
    delta = abs(new_total['nll'] - old_total['nll'])
    require(math.isfinite(delta) and delta <= TOLERANCE['aggregate_nll_abs'], 'Replay aggregate NLL exceeds original absolute tolerance')
    row_checks = []; layers = cfg['layer_types'].count('global'); B = cfg['block_size']; K = cfg['selected_complete_blocks']
    for row, old in zip(rows, old_rows):
        require(all(row[k] == old[k] for k in ROW_KEYS), 'Replay metadata/targets differ at window ' + str(old['window_index']))
        require(row['forward_calls'] == 1 and row['grad_enabled'] is False and row['model_training'] is False, 'Replay enabled gradients/training')
        n = old['input_tokens']; targets = old['loss_tokens']; require(targets > 0, 'Original fixed panel unexpectedly has zero targets')
        expected_kept = layers * sum(q + 1 - max(0, (q + 1) // B - K) * B for q in range(n))
        attention = row['attention_counts']
        require(attention['logical_kept_pairs'] == expected_kept and attention['logical_dense_causal_pairs'] == layers * n * (n + 1) // 2
                and attention['indexer_score_elements'] == attention['indexer_zero_score_visible_query_count'] == 0,
                'W per-window fixed local support differs')
        old_nll, new_nll = old['nll_sum'] / targets, row['nll_sum'] / targets
        error = abs(new_nll - old_nll); allowed = TOLERANCE['window_nll_abs']
        require(math.isfinite(error) and error <= allowed, 'Replay per-window NLL exceeds original absolute tolerance: ' + str(old['window_index']))
        row_checks.append({'window_index': old['window_index'], 'loss_tokens': targets, 'reference_nll': old_nll,
                           'replay_nll': new_nll, 'absolute_error': error, 'allowed_error': allowed,
                           'metadata_targets_exact': True, 'fixed_local_support_exact': True, 'passed': True})
    return {'passed': True, 'tolerance': TOLERANCE, 'aggregate_nll_absolute_error': delta,
            'max_window_nll_absolute_error': max(r['absolute_error'] for r in row_checks),
            'counts_exact': True, 'attention_counts_exact': True, 'window_checks': row_checks}


def execute(protocol_path, expected_sha):
    require(os.name == 'posix', 'ARM GPU replay execution requires Linux')
    path = resolve(protocol_path); require(sha(path) == expected_sha, 'Frozen migration master SHA differs')
    master = load(path); require(master.get('launch_allowed') is True, 'Migration gate launch not frozen')
    output = resolve(master['output_dir']); require(output.is_relative_to(ROOT / 'results') and not output.exists(), 'Fresh migration output required')
    output.mkdir(parents=True)
    counts = {'forward_attempts': 0, 'forward_calls': 0, 'committed_windows': 0, 'backward_calls': 0, 'optimizer_updates': 0}
    receipt = {'status': 'running', 'replay_passed': False, 'condition': common.CONDITION,
               'master_protocol_sha256': expected_sha, 'started_utc': datetime.now(timezone.utc).isoformat(),
               'counts': counts, 'scope': 'engineering_portability_replay', 'source_sha256': master['source_sha256'],
               'reference_runtime': master['reference_runtime'], 'reference_runtime_origin': master.get('reference_runtime_origin'),
               'tolerance': TOLERANCE, 'full_dev_dispatched': False, 'automatic_retry_or_training': False,
               'optimizer_or_rng_restored': False, 'gpu_lock_owner': 'enclosing bounded migration controller',
               'outer_timeout_seconds_required': master['hard_timeout_seconds']}
    write(output / 'replay-receipt.json', receipt); begin = time.monotonic()
    old_signal = signal.getsignal(signal.SIGTERM)
    def stop(signum, frame): raise InterruptedError('Migration replay interrupted')
    signal.signal(signal.SIGTERM, stop)
    try:
        plan = validate(master)
        require(platform.machine() == master['expected_machine'], 'Host machine architecture differs')
        common.configure_runtime()
        from src.babylm_hybrid.training import _runtime_signature
        runtime = _runtime_signature('cuda')
        require(runtime == master['expected_runtime'], 'Actual target GPU/runtime differs from frozen signature')
        runtime_check = validate_runtime(master['reference_runtime'], runtime)
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        require(driver == master['expected_driver_version'], 'Target driver differs')
        full = plan['full_eval_protocol']; local_eval.checkpoint_identity(full)
        child = derive_evaluation(master, plan, runtime)
        remaining = master['hard_timeout_seconds'] - 20 - (time.monotonic() - begin)
        require(remaining > 0, 'Replay setup exhausted outer deadline')
        child['max_wall_seconds'] = min(child['max_wall_seconds'], remaining)
        write(output / 'evaluation-protocol.json', child)
        receipt.update(verified_runtime=runtime, verified_runtime_sha256=canonical(runtime), runtime_comparison=runtime_check,
                       host_platform={'machine': platform.machine(), 'python': platform.python_version(), 'system': platform.system(), 'driver_version': driver},
                       reference_event_canonical_sha256=plan['reference_event_canonical_sha256'],
                       checkpoint_sha256=full['checkpoint_sha256'], checkpoint_protocol_sha256=full['checkpoint_protocol_sha256'],
                       window_indices=plan['indices'], window_indices_sha256=canonical(plan['indices']))
        write(output / 'replay-receipt.json', receipt)
        from scripts import run_babylm_checkpoint_eval_v0 as evaluator
        with local_eval.injected_factory(evaluator): summary = evaluator.run_evaluation(child)
        receipt['counts'] = summary['counts']
        rows = [json.loads(line) for line in (output / 'evaluation/windows.jsonl').read_text().splitlines()]
        receipt['numerical_comparison'] = compare_replay(plan['reference'], summary, rows, full['model_config'])
        require(sha(resolve(full['checkpoint_path'])) == full['checkpoint_sha256'] and
                sha(resolve(master['training_events_path'])) == master['training_events_sha256'], 'Original checkpoint/log mutated')
        require(_runtime_signature('cuda') == runtime, 'Runtime changed during replay')
        receipt.update(status='passed', replay_passed=True, engineering_forward_calls=48, scientific_forward_calls=0,
                       checkpoint_unchanged=True, original_log_unchanged=True, actual_total=summary['total'],
                       source_and_dependency=summary['transformers_gdn_source'],
                       data_fingerprint=summary['data_fingerprint'], data_fingerprint_sha256=canonical(summary['data_fingerprint']),
                       evaluator_protocol_sha256=canonical(child), evaluator_summary_sha256=sha(output / 'evaluation/summary.json'),
                       per_window_jsonl_sha256=sha(output / 'evaluation/windows.jsonl'))
    except BaseException as e:
        receipt.update(status='failed', replay_passed=False, error_type=type(e).__name__, error=str(e), traceback=traceback.format_exc())
        if (output / 'evaluation/summary.json').exists(): receipt['counts'] = load(output / 'evaluation/summary.json')['counts']
        receipt.update(engineering_forward_calls=receipt['counts']['forward_calls'], scientific_forward_calls=0)
    finally:
        receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_wall_seconds=time.monotonic() - begin)
        write(output / 'replay-receipt.json', receipt); signal.signal(signal.SIGTERM, old_signal)
    print(json.dumps({k: receipt[k] for k in ['status', 'replay_passed', 'counts', 'elapsed_wall_seconds']}, allow_nan=False))
    return 0 if receipt['replay_passed'] else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True); p.add_argument('--protocol-sha256'); p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.execute:
        require(bool(args.protocol_sha256), 'Explicit master SHA required')
        return execute(args.protocol, args.protocol_sha256)
    plan = validate(load(resolve(args.protocol)))
    print(json.dumps({'status': 'static_validation_passed', 'reference_event_sha256': plan['reference_event_canonical_sha256'],
                      'required_engineering_forward_calls_if_executed': 48, 'model_calls': 0, 'cuda_calls': 0}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
