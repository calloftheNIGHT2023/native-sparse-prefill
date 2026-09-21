"""One bounded, append-free W scoring remainder; never a training resume.

The original timed-out 12,402-window run stays immutable. Only the remaining
6,390 independent reset windows are scored, using the frozen W factory/scorer.
An enclosing controller owns the GPU lock and hard process-group deadline.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_babylm_local_train_v0 as common
from scripts import run_babylm_local_eval_v0 as local_eval
require, resolve, sha, load, write, canonical = common.require, common.resolve, common.sha, common.load, common.write, common.canonical
SELF = 'scripts/run_babylm_local_eval_remainder_v0.py'
FULL_SHA = '0c5e625ba993a3a0b51b12d167fe64d5a63a364f5cf31d99f7ff98e5f49d7d4d'
PREFIX_WINDOWS_SHA = 'c23183ba0f66ef70d359cb970561876a5023c652d90ab2328a1274eccf3dd959'
PREFIX_SUMMARY_SHA = '42e3f00f974f931f8793db287b0ed93ee4e4bcda7c2be1aac1eb4239b89ea990'
START, STOP = 12402, 18792
OUTPUT = 'results/babylm-w-arm-remainder-20260921-v0'
BINS = ((1, 256), (257, 512), (513, 1024), (1025, 2048))
PREFIX_COUNTS = {'windows': 12402, 'input_tokens': 15739613, 'loss_tokens': 15727211, 'word_exposures': 9320813}
REMAINDER_COUNTS = {'windows': 6390, 'input_tokens': 1697921, 'loss_tokens': 1691531, 'word_exposures': 1100149}
FULL_COUNTS = {k: PREFIX_COUNTS[k] + REMAINDER_COUNTS[k] for k in PREFIX_COUNTS}


def rows(path):
    raw = path.read_bytes()
    require(raw.endswith(b'\n'), 'Partial JSONL tail cannot be accepted')
    return [json.loads(line) for line in raw.splitlines()]


def finite(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), 'Nonfinite ' + name)
    return value


def scalar_metrics(total, count):
    finite(total, 'NLL sum')
    nll = total / count if count else None
    ppl = math.exp(nll) if nll is not None else None
    require(ppl is None or math.isfinite(ppl), 'PPL overflow')
    return {'loss_tokens': count, 'nll_sum': total, 'nll': nll, 'ppl': ppl}


def aggregate(selected, source='all'):
    attention = Counter()
    for row in selected: attention.update(row['attention_counts'])
    result = {'source': source, 'windows': len(selected), 'forward_calls': len(selected),
              'zero_target_windows': sum(row['loss_tokens'] == 0 for row in selected),
              'word_exposures': sum(row['word_exposures'] for row in selected),
              'input_tokens': sum(row['input_tokens'] for row in selected),
              'attention_counts': dict(attention)}
    result.update(scalar_metrics(math.fsum(row['nll_sum'] for row in selected), sum(row['loss_tokens'] for row in selected)))
    dense = attention.get('logical_dense_causal_pairs', 0)
    result['logical_retained_fraction'] = attention.get('logical_kept_pairs', 0) / dense if dense else None
    return result


def position_aggregate(selected):
    return {f'{lo}-{hi}': scalar_metrics(math.fsum(row['query_history_bins'][f'{lo}-{hi}']['nll_sum'] for row in selected),
                                        sum(row['query_history_bins'][f'{lo}-{hi}']['loss_tokens'] for row in selected))
            for lo, hi in BINS}


def load_layout(full):
    """Hash original data and read the immutable integer index; no torch/model."""
    import numpy as np
    path = resolve(full['dev_manifest'])
    require(sha(path) == full['dev_manifest_sha256'], 'Fixed development manifest differs')
    manifest = load(path)
    require(manifest['total_windows'] == STOP and manifest['max_window_tokens'] == 2048, 'Fixed dev shape differs')
    for artifact in manifest['artifacts']:
        target = (path.parent / artifact['path']).resolve()
        require(target.parent == path.parent and sha(target) == artifact['sha256'], 'Dev artifact differs')
    for source in manifest['source_summaries']:
        require(sha(resolve(source['token_ids_path_relative_to_project'])) == source['token_ids_sha256'], 'Dev token stream differs')
    index = np.load(path.parent / 'windows.u64.npy', mmap_mode='r', allow_pickle=False)
    require(index.shape == (STOP, len(manifest['columns'])) and index.dtype == np.dtype('uint64'), 'Dev index schema differs')
    columns = {name: pos for pos, name in enumerate(manifest['columns'])}
    sources = {source['source_index']: source['source'] for source in manifest['source_summaries']}
    return manifest, index, columns, sources


def audit_rows(observed, indices, layout, cfg):
    manifest, index, columns, sources = layout
    require([r['window_index'] for r in observed] == list(indices), 'Missing, repeated, or reordered window')
    layers = cfg['layer_types'].count('global'); block = cfg['block_size']; selected = cfg['selected_complete_blocks']
    for row in observed:
        expected = index[row['window_index']]
        for key in ('source_index', 'segment_index_in_source', 'source_token_start', 'source_token_end', 'word_exposures', 'input_tokens'):
            require(type(row[key]) is int and row[key] == int(expected[columns[key]]), 'Window index metadata differs: ' + key)
        n, targets = row['input_tokens'], row['loss_tokens']
        require(targets == int(expected[columns['next_token_loss_positions']]) == n - 1, 'Target count differs')
        require(row['source'] == sources[row['source_index']] and row['forward_calls'] == 1
                and row['grad_enabled'] is False and row['model_training'] is False, 'Source/mode/call identity differs')
        require(row.get('routing_policy') == 'learned', 'Frozen scorer compatibility label differs')
        total = finite(row['nll_sum'], 'row NLL sum')
        if targets:
            require(math.isclose(finite(row['nll'], 'row NLL') * targets, total, rel_tol=1e-12, abs_tol=1e-8), 'Row LM reduction differs')
            require(math.isclose(finite(row['ppl'], 'row PPL'), math.exp(row['nll']), rel_tol=1e-12), 'Row PPL differs')
        else:
            require(total == 0 and row['nll'] is None and row['ppl'] is None, 'Zero-target row must not contain fabricated NLL')
        parts = row['query_history_bins']
        require(set(parts) == {f'{lo}-{hi}' for lo, hi in BINS}, 'Frozen position bins differ')
        for lo, hi in BINS:
            part = parts[f'{lo}-{hi}']
            require(part['loss_tokens'] == max(0, min(targets, hi) - lo + 1), 'Position denominator differs')
            finite(part['nll_sum'], 'position sum')
            require(part['loss_tokens'] > 0 or part['nll_sum'] == 0, 'Empty position bin has loss')
        require(math.isclose(math.fsum(p['nll_sum'] for p in parts.values()), total, rel_tol=1e-5, abs_tol=1e-5), 'Position/LM reductions disagree')
        require(row['window_length_bin'] == next(f'{lo}-{hi}' for lo, hi in BINS if lo <= n <= hi), 'Length bin differs')
        attention = row['attention_counts']
        require(all(type(v) is int and v >= 0 for v in attention.values()), 'Attention counts must be nonnegative integers')
        require(attention['logical_dense_causal_pairs'] == layers * n * (n + 1) // 2, 'Dense support count differs')
        kept = sum(q + 1 - max(0, (q + 1) // block - selected) * block for q in range(n)) * layers
        require(attention['logical_kept_pairs'] == kept and attention['indexer_score_elements'] == 0
                and attention['indexer_zero_score_visible_query_count'] == 0, 'W local support/indexer identity differs')
    return aggregate(observed)


def audit_summary(summary, observed, full, expected_counts, complete):
    n = len(observed); counts = summary['counts']
    require(counts['forward_attempts'] == counts['forward_calls'] == counts['committed_windows'] == n,
            'In-flight/uncommitted/extra forward cannot be silently omitted')
    require(counts['backward_calls'] == counts['optimizer_updates'] == summary['backward_calls'] == summary['optimizer_updates'] == 0, 'Scoring performed a backward/update')
    require(summary['observed_window_indices'] == [r['window_index'] for r in observed], 'Summary committed IDs differ')
    require(summary['scope'] == 'scientific_evaluation' and summary['runtime'] == full['expected_runtime'], 'Original scientific/runtime identity differs')
    require(summary['source_hashes'] == full['expected_source_hashes'] and summary['transformers_gdn_source']['sha256'] == full['transformers_gdn_source_sha256'], 'Scientific source/GDN identity differs')
    require(summary['checkpoint']['sha256'] == full['checkpoint_sha256']
            and summary['checkpoint']['training_protocol_sha256'] == full['checkpoint_protocol_sha256'], 'Scored model identity differs')
    require(summary['data_fingerprint']['manifest_sha256'] == full['dev_manifest_sha256'], 'Scored development data differs')
    if complete:
        require(summary['status'] == 'evaluation_complete' and summary['partial_metrics_only'] is False, 'Remainder did not complete')
        require(summary['requested_windows'] == STOP - START and summary['window_indices'] == list(range(START, STOP)), 'Remainder requested a different subset')
    else:
        require(summary['status'] == 'evaluation_failed' and summary['error_type'] == 'TimeoutError'
                and summary['partial_metrics_only'] is True and summary['requested_windows'] == STOP
                and summary['window_indices'] == list(range(STOP)), 'Original terminal timeout evidence differs')
    calculated = aggregate(observed)
    for key, value in expected_counts.items(): require(calculated[key] == value, 'Frozen denominator differs: ' + key)
    for key in ('windows', 'word_exposures', 'input_tokens', 'loss_tokens', 'forward_calls', 'zero_target_windows', 'attention_counts'):
        require(calculated[key] == summary['total'][key], 'Summary raw count differs: ' + key)
    require(counts['submitted_input_tokens'] == calculated['input_tokens'] and counts['submitted_word_exposures'] == calculated['word_exposures'], 'Submitted accounting differs')
    for key in ('nll_sum', 'nll', 'ppl'):
        require(math.isclose(calculated[key], summary['total'][key], rel_tol=1e-12, abs_tol=1e-8), 'Summary raw LM metric differs: ' + key)
    for name in summary['per_source']:
        actual = aggregate([r for r in observed if r['source'] == name], name)
        for key in ('windows', 'word_exposures', 'input_tokens', 'loss_tokens', 'attention_counts'):
            require(actual[key] == summary['per_source'][name][key], 'Per-source denominator differs')
        require(math.isclose(actual['nll_sum'], summary['per_source'][name]['nll_sum'], rel_tol=1e-12, abs_tol=1e-8), 'Per-source LM reduction differs')
    pos = position_aggregate(observed)
    for name, value in pos.items():
        old = summary['position_diagnostics']['query_history_bins'][name]
        require(old['loss_tokens'] == value['loss_tokens'] and math.isclose(old['nll_sum'], value['nll_sum'], rel_tol=1e-12, abs_tol=1e-8), 'Summary position aggregate differs')
    return calculated


def validate(master):
    require(master['schema_version'] == 1 and master['condition'] == common.CONDITION, 'Explicit W remainder identity required')
    require(0 < master['max_wall_seconds'] <= 3200 and master['max_wall_seconds'] + 30 <= master['hard_timeout_seconds'] <= 3300, 'New bounded remainder wall limits required')
    require(master['output_dir'] == OUTPUT, 'Only the fixed new remainder directory may be written')
    for key in ('source_full_protocol_path', 'prefix_windows_path', 'prefix_summary_path', 'output_dir'):
        require('\\' not in master[key], 'Deployment paths must use POSIX separators')
    fp = resolve(master['source_full_protocol_path'])
    require(master['source_full_protocol_sha256'] == sha(fp) == FULL_SHA, 'Original frozen full protocol differs')
    full = load(fp); audit = local_eval.validate(full)
    require(master['source_sha256'] == dict(full['expected_source_hashes'], **{SELF: sha(__file__)}), 'Only original26 sources plus new administrative wrapper are allowed')
    for name, digest in master['source_sha256'].items(): require(sha(resolve(name)) == digest, 'Pinned source differs: ' + name)
    wp, sp = resolve(master['prefix_windows_path']), resolve(master['prefix_summary_path'])
    require(master['prefix_windows_sha256'] == sha(wp) == PREFIX_WINDOWS_SHA
            and master['prefix_summary_sha256'] == sha(sp) == PREFIX_SUMMARY_SHA, 'Original terminal prefix bytes differ')
    prefix, summary = rows(wp), load(sp)
    require(summary['windows_jsonl_sha256'] == PREFIX_WINDOWS_SHA, 'Original summary/log binding differs')
    layout = load_layout(full)
    audit_rows(prefix, range(START), layout, full['model_config'])
    audit_summary(summary, prefix, full, PREFIX_COUNTS, complete=False)
    return {'full': full, 'training_audit': audit, 'prefix': prefix, 'prefix_summary': summary, 'layout': layout}


def derive_child(master, full):
    child = copy.deepcopy(full)
    child.update(window_indices=list(range(START, STOP)), output_dir=OUTPUT + '/evaluation',
                 max_wall_seconds=master['max_wall_seconds'], hard_timeout_seconds=master['hard_timeout_seconds'],
                 administrative_scoring_remainder=True, training_resume=False,
                 original_timed_out_protocol_sha256=FULL_SHA, prefix_windows_sha256=PREFIX_WINDOWS_SHA,
                 original_absolute_deadline_utc=full.get('absolute_deadline_utc'))
    child.pop('absolute_deadline_utc', None)
    if 'absolute_deadline_utc' in master: child['absolute_deadline_utc'] = master['absolute_deadline_utc']
    # Scientific source26, checkpoint, scoring and runtime remain byte-for-byte values.
    return child


def combined_result(plan, observed, remainder_summary):
    union = plan['prefix'] + observed
    total = audit_rows(union, range(STOP), plan['layout'], plan['full']['model_config'])
    require(all(total[k] == v for k, v in FULL_COUNTS.items()), 'Union does not cover original full dev')
    names = list(plan['layout'][3].values())
    return {'status': 'complete_local_dev_union_audited', 'condition': common.CONDITION,
            'scope': 'scientific_evaluation_union_of_two_disjoint_runs', 'single_uninterrupted_run': False,
            'total': total, 'per_source': {name: aggregate([r for r in union if r['source'] == name], name) for name in names},
            'position_diagnostics': {'query_history_bins': position_aggregate(union),
                'per_source_query_history_bins': {name: position_aggregate([r for r in union if r['source'] == name]) for name in names},
                'extra_model_forwards': 0},
            'observed_window_indices_sha256': canonical(list(range(STOP))),
            'cumulative_scientific_forward_calls': STOP, 'new_scientific_forward_calls': STOP - START,
            'backward_calls': 0, 'optimizer_updates': 0,
            'prefix_status': plan['prefix_summary']['status'], 'prefix_error_type': plan['prefix_summary']['error_type'],
            'prefix_elapsed_wall_seconds': plan['prefix_summary']['elapsed_wall_seconds'],
            'remainder_elapsed_wall_seconds': remainder_summary['elapsed_wall_seconds'],
            'timing_scope': 'Two scoring intervals; no sparse-training speed or statistical equivalence claim.'}


def execute(path, expected_sha):
    path = resolve(path); require(sha(path) == expected_sha, 'Frozen remainder master differs')
    master = load(path); require(master.get('launch_allowed') is True, 'Remainder launch not frozen')
    output = resolve(master['output_dir'])
    require(master['output_dir'] == OUTPUT and output.is_relative_to(ROOT / 'results') and not output.exists(), 'Fresh fixed remainder output required')
    output.mkdir(parents=True)
    receipt = {'status': 'running', 'condition': common.CONDITION, 'master_protocol_sha256': expected_sha,
               'started_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'administrative_scoring_remainder',
               'prefix_windows_sha256': master['prefix_windows_sha256'], 'prefix_summary_sha256': master['prefix_summary_sha256'],
               'prefix_windows_path': master['prefix_windows_path'], 'prefix_summary_path': master['prefix_summary_path'],
               'source_sha256': master['source_sha256'], 'new_scientific_forward_calls': 0,
               'counts': {'forward_attempts': 0, 'forward_calls': 0, 'committed_windows': 0, 'backward_calls': 0, 'optimizer_updates': 0},
               'training_resume': False, 'automatic_retry': False, 'old_run_extended_or_written': False,
               'hard_timeout_seconds_required': master['hard_timeout_seconds'], 'gpu_lock_owner': 'enclosing bounded controller'}
    receipt_path = output / 'remainder-receipt.json'; write(receipt_path, receipt)
    begin = time.monotonic(); previous = signal.getsignal(signal.SIGTERM)
    def stop(signum, frame): raise InterruptedError('W remainder stopped by enclosing controller')
    signal.signal(signal.SIGTERM, stop)
    try:
        plan = validate(master); full = plan['full']; child = derive_child(master, full)
        receipt['checkpoint_sha256'] = full['checkpoint_sha256']
        receipt['checkpoint_protocol_sha256'] = full['checkpoint_protocol_sha256']
        local_eval.checkpoint_identity(full)
        remaining = master['hard_timeout_seconds'] - (time.monotonic() - begin) - 30
        require(remaining > 0, 'Setup consumed the remainder deadline')
        child['max_wall_seconds'] = min(child['max_wall_seconds'], remaining)
        write(output / 'evaluation-protocol.json', child)
        from scripts import run_babylm_checkpoint_eval_v0 as evaluator
        with local_eval.injected_factory(evaluator): summary = evaluator.run_evaluation(child)
        receipt['counts'] = summary['counts']; receipt['new_scientific_forward_calls'] = summary['counts']['forward_calls']
        observed = rows(output / 'evaluation/windows.jsonl')
        audit_rows(observed, range(START, STOP), plan['layout'], full['model_config'])
        audit_summary(summary, observed, full, REMAINDER_COUNTS, complete=True)
        require(summary['data_fingerprint'] == plan['prefix_summary']['data_fingerprint'], 'Original/remainder data fingerprints differ')
        for key, expected in [('prefix_windows_path', PREFIX_WINDOWS_SHA), ('prefix_summary_path', PREFIX_SUMMARY_SHA)]:
            require(sha(resolve(master[key])) == expected, 'Original evidence was modified')
        require(sha(resolve(full['checkpoint_path'])) == full['checkpoint_sha256'], 'Original checkpoint was modified')
        for name, digest in master['source_sha256'].items(): require(sha(resolve(name)) == digest, 'Source changed during evaluation')
        combined = combined_result(plan, observed, summary); write(output / 'combined-summary.json', combined)
        receipt.update(status='complete_local_remainder_and_union_audited', total=summary['total'], combined_total=combined['total'],
                       cumulative_scientific_forward_calls=STOP, backward_calls=0, optimizer_updates=0,
                       original_evidence_unchanged=True, summary_sha256=sha(output / 'evaluation/summary.json'),
                       windows_jsonl_sha256=sha(output / 'evaluation/windows.jsonl'),
                       combined_summary_sha256=sha(output / 'combined-summary.json'), evaluator_protocol_sha256=canonical(child),
                       verified_runtime=summary['runtime'], observed_window_indices_sha256=canonical(list(range(START, STOP))))
    except BaseException as error:
        receipt.update(status='local_remainder_failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if (output / 'evaluation/summary.json').exists():
            failed = load(output / 'evaluation/summary.json')
            receipt['counts'] = failed['counts']; receipt['new_scientific_forward_calls'] = failed['counts']['forward_calls']
    finally:
        receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_wall_seconds=time.monotonic() - begin)
        write(receipt_path, receipt); signal.signal(signal.SIGTERM, previous)
    print(json.dumps({k: receipt.get(k) for k in ('status', 'counts', 'combined_total')}, allow_nan=False))
    return 0 if receipt['status'] == 'complete_local_remainder_and_union_audited' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True); parser.add_argument('--protocol-sha256'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        require(bool(args.protocol_sha256), 'Execution requires explicit master SHA')
        return execute(args.protocol, args.protocol_sha256)
    validate(load(resolve(args.protocol)))
    print(json.dumps({'status': 'static_validation_passed', 'fixed_remainder_windows': STOP - START, 'model_calls': 0, 'cuda_calls': 0}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
