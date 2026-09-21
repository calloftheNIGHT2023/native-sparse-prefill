"""Independently audit the two-segment W full-dev union. No model/remote calls.

Run only after an immutable completed remainder collection exists:
  python audit_union.py --manifest PATH --manifest-sha256 SHA
Default output is metrics-audit.json beside this script; existing files are never overwritten.
The timed-out prefix remains immutable and its cost/time is retained separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[2]
PREFIX_MANIFEST = 'logs/babylm-w-arm-full-dev-backup-20260921-v0/manifest-20260921T060307.585991Z-dcee3b9a.json'
PREFIX_MANIFEST_SHA = '5a528008bdece2e7b6c4bc6c7fb1866fc0ba2739873e8efe38d2e56ec10c981a'
PREFIX_AUDIT = 'results/babylm-stage-w-partial-20260921/metrics-audit.json'
PREFIX_AUDIT_SHA = '3a3faae58570e8bd74d699000daf99b12f147d0eb753ce9638d452e3a1cf12c4'
COMPLETION_AUDIT = 'logs/babylm-w-original-completion-audit-20260921.json'
COMPLETION_AUDIT_SHA = '611e45bfd07976dfbeba96d8f0559d7ed9c595712d8b00d1fb5f22fa06bf6f3c'
REPLAY_AUDIT = 'logs/babylm-w-arm-replay-independent-audit-20260921.json'
REPLAY_AUDIT_SHA = '321e800526b9d2e290ec5ee0056afa7d5673bca7bde2dd837ebfc058540e0612'
FINAL_SHA = '854b912cb76e889abd6c3fb7ef101c004b42576cc01425b6dde19033b790ab39'
TRAINING_PROTOCOL_SHA = 'e59de1c0d9133490876dd31f8f32694bcc82a344b96225f8e0be35e039b01389'
FIELDS = ('window_index', 'source_index', 'source', 'segment_index_in_source', 'source_token_start',
          'source_token_end', 'word_exposures', 'input_tokens', 'loss_tokens')
BINS = {'1-256': (1, 256), '257-512': (257, 512), '513-1024': (513, 1024), '1025-2048': (1025, 2048)}
EXPECTED = {'windows': 18792, 'input_tokens': 17437534, 'loss_tokens': 17418742, 'word_exposures': 10420962}
PREFIX = {'windows': 12402, 'input_tokens': 15739613, 'loss_tokens': 15727211, 'word_exposures': 9320813}


def read_json(path):
    def bad(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=bad)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def validate_partition(prefix, remainder, total=18792, cut=12402):
    """Exact order, contiguous coverage and no repeats; supports tiny metadata fixtures."""
    require(len(prefix) == cut and len(remainder) == total-cut, 'Wrong prefix/remainder lengths')
    left = [r['window_index'] for r in prefix]
    right = [r['window_index'] for r in remainder]
    require(left == list(range(cut)), 'Prefix window sequence differs')
    require(right == list(range(cut, total)), 'Remainder has a gap, duplicate, overlap or reordered window')
    require(len(set(left+right)) == total, 'Union does not contain every window exactly once')


def metric(rows):
    targets = sum(r['loss_tokens'] for r in rows)
    nll_sum = math.fsum(r['nll_sum'] for r in rows)
    return {'records': len(rows), 'loss_tokens': targets, 'nll_sum': nll_sum,
            'nll': nll_sum/targets if targets else None,
            'ppl': math.exp(nll_sum/targets) if targets else None}


def aggregate(rows):
    sources, positions = defaultdict(list), defaultdict(list)
    source_positions = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sources[row['source']].append(row)
        for label, cell in row['query_history_bins'].items():
            positions[label].append(cell)
            source_positions[row['source']][label].append(cell)
    return {'total': metric(rows), 'per_source': {k: metric(v) for k, v in sources.items()},
            'position': {k: metric(v) for k, v in positions.items()},
            'per_source_position': {s: {k: metric(v) for k, v in bins.items()} for s, bins in source_positions.items()}}


def difference(a, b):
    return {'nll_difference': a['nll']-b['nll'], 'ppl_relative_difference': math.expm1(a['nll']-b['nll'])}


class Audit:
    def __init__(self):
        self.report = {'schema_version': 1, 'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                       'checks': 0, 'failures': [], 'input_files': {},
                       'analysis_counts': {'model_calls': 0, 'forward_calls': 0, 'backward_calls': 0,
                                           'optimizer_updates': 0, 'cuda_calls': 0, 'remote_calls': 0, 'checkpoint_loads': 0}}

    def check(self, value, label):
        self.report['checks'] += 1
        if not bool(value):
            self.report['failures'].append(label)

    def file(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), 'Input path escapes research project')
        actual = sha(path)
        if expected is not None:
            require(actual == expected, 'Input SHA mismatch: ' + path.name)
        self.report['input_files'][path.relative_to(ROOT).as_posix()] = {'sha256': actual, 'bytes': path.stat().st_size}
        return path

    def collection(self, path, expected):
        path = self.file(path, expected)
        data = read_json(path)
        require(data['status'] == 'collection_complete_all_files_sha_verified', 'Collection is not complete')
        result = {}
        for receipt in data['receipts']:
            require(receipt['status'] in ('reused_sha_verified', 'downloaded_sha_verified'), 'Unverified collected artifact')
            local = self.file(path.parent/receipt['local_relative_path'], receipt['sha256'])
            require(local.is_relative_to(path.parent) and local.stat().st_size == receipt['size_bytes'], 'Collected path/size differs')
            require(receipt['relative_path'] not in result, 'Duplicate receipt name')
            result[receipt['relative_path']] = local
        return result

    def rows(self, path):
        raw = Path(path).read_bytes()
        require(raw.endswith(b'\n'), 'Truncated JSONL record')
        return [json.loads(row, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))) for row in raw.splitlines()]

    def segment(self, rows, summary, first, count):
        self.check(len(rows) == count and [r['window_index'] for r in rows] == list(range(first, first+count)), 'segment exact ID range')
        totals = {k: sum(r[k] for r in rows) for k in ('input_tokens', 'loss_tokens', 'word_exposures')}
        counts = {'forward_attempts': count, 'forward_calls': count, 'committed_windows': count,
                  'submitted_input_tokens': totals['input_tokens'], 'submitted_word_exposures': totals['word_exposures'],
                  'backward_calls': 0, 'optimizer_updates': 0}
        self.check(summary['counts'] == counts, 'segment actual F/tokens/zero B and updates')
        self.check(summary['observed_window_indices'] == [r['window_index'] for r in rows], 'segment committed indices match summary')
        self.check(all(totals[k] == summary['total'][k] for k in totals), 'segment target ledger matches summary')
        mm = metric(rows)
        self.check(abs(mm['nll']-summary['total']['nll']) <= 1e-12, 'segment raw token weighted NLL')
        return counts


def unique(files, suffix):
    candidates = [path for name, path in files.items() if name.endswith(suffix)]
    require(len(candidates) == 1, 'Expected exactly one artifact ending in ' + suffix)
    return candidates[0]


def run(a, manifest, manifest_sha):
    prefix_files = a.collection(ROOT/PREFIX_MANIFEST, PREFIX_MANIFEST_SHA)
    remainder_files = a.collection(manifest, manifest_sha)
    pre = 'results/babylm-w-arm-full-dev-20260921-v0/'
    pw = prefix_files[pre+'windows.jsonl']; ps = read_json(prefix_files[pre+'summary.json'])
    pp = read_json(prefix_files[pre+'protocol.json'])
    rw = unique(remainder_files, '/evaluation/windows.jsonl'); rs = read_json(unique(remainder_files, '/evaluation/summary.json'))
    rp = read_json(unique(remainder_files, '/evaluation/protocol.json'))
    receipt = read_json(unique(remainder_files, '/remainder-receipt.json'))
    combined_path = unique(remainder_files, '/combined-summary.json')
    combined = read_json(combined_path)
    launch = read_json(unique(remainder_files, '/launch.json'))
    require(launch['protocol'] == 'configs/babylm-w-arm-remainder-20260921-v0.json', 'Wrong remainder master path')
    master_path = a.file(ROOT/launch['protocol'], launch['protocol_sha256'])
    master = read_json(master_path)
    a.check(sha(master_path) == receipt['master_protocol_sha256'], 'frozen remainder master/launcher/receipt binding')
    a.check(master['source_sha256'] == receipt['source_sha256'], 'remainder master source pins')
    a.check(receipt['source_sha256']['scripts/run_babylm_local_eval_remainder_v0.py']
            == 'bdad87ece451ffd2831cd862c74d9e49f99430cdae75e49d492001c1fb37624d', 'frozen administrative remainder wrapper')
    require(receipt['status'] == 'complete_local_remainder_and_union_audited', 'Remainder wrapper did not establish completion')
    require(ps['status'] == 'evaluation_failed' and ps['error_type'] == 'TimeoutError' and ps['partial_metrics_only'] is True,
            'Original prefix timeout record was changed or misrepresented')
    require(rs['status'] == 'evaluation_complete' and rs['partial_metrics_only'] is False, 'Remainder incomplete')
    require(rs['requested_windows'] == 6390 and rs['full_manifest_requested'] is False, 'Remainder must request only6390 fixed missing windows')
    left, right = a.rows(pw), a.rows(rw)
    validate_partition(left, right)
    a.check(sha(pw) == ps['windows_jsonl_sha256'] and sha(rw) == rs['windows_jsonl_sha256'], 'both raw SHA bindings')
    a.check(canonical(pp) == ps['protocol_sha256'] and canonical(rp) == rs['protocol_sha256'], 'both scorer protocol bindings')
    a.check(receipt['prefix_windows_sha256'] == sha(pw)
            and receipt['prefix_summary_sha256'] == sha(prefix_files[pre+'summary.json']), 'wrapper exact original prefix evidence')
    a.check(receipt['summary_sha256'] == sha(unique(remainder_files, '/evaluation/summary.json'))
            and receipt['windows_jsonl_sha256'] == sha(rw)
            and receipt['combined_summary_sha256'] == sha(combined_path)
            and receipt['evaluator_protocol_sha256'] == canonical(rp), 'wrapper new raw and union SHA bindings')
    a.check(receipt['training_resume'] is False and receipt['automatic_retry'] is False
            and receipt['old_run_extended_or_written'] is False and receipt['original_evidence_unchanged'] is True,
            'explicit separate evaluation-only continuation')
    a.check(rp['window_indices'] == list(range(12402, 18792)), 'frozen remainder membership')
    old_audit = read_json(a.file(ROOT/COMPLETION_AUDIT, COMPLETION_AUDIT_SHA))
    partial_audit = read_json(a.file(ROOT/PREFIX_AUDIT, PREFIX_AUDIT_SHA))
    replay_audit = read_json(a.file(ROOT/REPLAY_AUDIT, REPLAY_AUDIT_SHA))
    a.check(partial_audit['status'] == 'passed_partial_only_metrics_audit' and replay_audit['status'] == 'passed_independent_raw48_ARM_replay_audit', 'prior original prefix and migration gates')
    for summary in (ps, rs):
        a.check(summary['checkpoint']['sha256'] == FINAL_SHA == old_audit['final_model']['sha256'], 'same unique W final1413')
        a.check(summary['checkpoint']['training_protocol_sha256'] == TRAINING_PROTOCOL_SHA, 'original W training protocol')
        a.check(summary['checkpoint']['source_hashes_recorded_by_training'] == old_audit['full_checkpoint']['source_hashes'], 'unchanged original W checkpoint source map')
        a.check(summary['optimizer_updates'] == summary['backward_calls'] == 0, 'evaluation only no model training')
    a.check(old_audit['final_model']['marker'] == [4, 64] and old_audit['final_model']['matches_full_checkpoint_all_tensors'] is True,
            'persistent W checkpoint marker provenance')
    a.check(ps['runtime'] == rs['runtime'] and rs['runtime']['gpu']['name'] == 'NVIDIA GB300', 'same runtime for both evaluation segments')
    a.check(ps['data_fingerprint'] == rs['data_fingerprint'], 'identical frozen dev fingerprint')
    a.check(datetime.fromisoformat(rs['started_utc']) > datetime.fromisoformat(ps['completed_utc']), 'separate later continuation segment')
    if 'hard_deadline_utc' in launch:
        a.check(datetime.fromisoformat(rs['completed_utc']) < datetime.fromisoformat(launch['hard_deadline_utc']),
                'remainder finished before newly authorized finite deadline')
    a.check(rp['checkpoint_sha256'] == pp['checkpoint_sha256'] == FINAL_SHA and rp['model_config'] == pp['model_config'], 'same model weights and scientific architecture')
    for key in ('dev_manifest', 'dev_manifest_sha256', 'position_diagnostics', 'enable_routing_diagnostics', 'mode', 'dtype', 'torch_num_threads'):
        a.check(rp[key] == pp[key], 'unchanged evaluation field ' + key)
    for source, digest in rp['expected_source_hashes'].items():
        a.file(ROOT/source, digest)
    a.check(rp['expected_source_hashes']['src/babylm_hybrid/local_attention_v0.py'] == '087813356a96338232bbaf79d331463bccc322f383f88c0246d185d4ff111e8c', 'original fixed-local implementation')
    left_counts = a.segment(left, ps, 0, 12402); right_counts = a.segment(right, rs, 12402, 6390)
    a.check(receipt['counts'] == right_counts and receipt['new_scientific_forward_calls'] == 6390
            and receipt['cumulative_scientific_forward_calls'] == 18792
            and receipt['backward_calls'] == receipt['optimizer_updates'] == 0, 'wrapper new/cumulative physical calls')
    a.check(receipt['verified_runtime'] == rs['runtime'] and receipt['checkpoint_sha256'] == FINAL_SHA
            and receipt['checkpoint_protocol_sha256'] == TRAINING_PROTOCOL_SHA, 'wrapper actual checkpoint/runtime')
    for source, digest in receipt['source_sha256'].items():
        a.file(ROOT/source, digest)
    for key, value in PREFIX.items():
        actual = len(left) if key == 'windows' else sum(r[key] for r in left)
        a.check(actual == value, 'unchanged original prefix '+key)
    joined = left+right
    for key, value in EXPECTED.items():
        actual = len(joined) if key == 'windows' else sum(r[key] for r in joined)
        a.check(actual == value, 'complete union '+key)
    support = [0]
    for n in range(1, 2049): support.append(support[-1]+4*min(n//4, 64)+n%4)
    attention = defaultdict(int)
    for row in joined:
        n, targets = row['input_tokens'], row['loss_tokens']
        a.check(1 <= n <= 2048 and targets == n-1, 'valid target count')
        expect = {'valid_tokens': 3*n, 'aux_loss_query_count': 3*targets, 'segments': 3,
                  'logical_dense_causal_pairs': 3*n*(n+1)//2, 'logical_kept_pairs': 3*support[n],
                  'allocated_main_score_elements': 18*n*n, 'indexer_score_elements': 0,
                  'aux_nonempty_support_query_count': 0, 'indexer_zero_score_visible_query_count': 0}
        a.check(row['attention_counts'] == expect, 'independent full union local support '+str(row['window_index']))
        for key, value in row['attention_counts'].items(): attention[key] += value
        a.check(row['grad_enabled'] is False and row['model_training'] is False and row['forward_calls'] == 1, 'read-only evaluation row')
        if targets:
            a.check(math.isfinite(row['nll']) and abs(row['nll']-row['nll_sum']/targets) < 1e-12, 'finite target weighted window NLL')
        else:
            a.check(row['nll'] is None and row['nll_sum'] == 0, 'zero target window preserved')
        for name, (lo, hi) in BINS.items():
            a.check(row['query_history_bins'][name]['loss_tokens'] == max(0, min(targets, hi)-lo+1), 'exact position denominator')
    base = ROOT/'logs/babylm-optimization-stage-a-20260920-v0-observation-20260920T045631Z/raw/results/babylm-optimization-stage-a-20260920-v0'
    baselines = {'D': (base/'full-dev-dense/windows.jsonl', 'd2fa9e73e52d4b86ea77d725f6a2ad1a9bc0365bd734294f6d50a182c3dec50c'),
                 'E': (base/'full-dev-sparse/windows.jsonl', '5ad78902705d2f7e1dd3da11305718f856bf9c60aa811449ca71dd7c016849e5'),
                 'F': (ROOT/'logs/babylm-stage-c2-backup-20260920-v0/results/babylm-stage-c2-scale-recovery-20260920-v0/full-dev/evaluation/windows.jsonl', '4ad4589b3943cacab88be4afcb2cb7dc7c4af7b0628b1eacd3a1d4eeaa3d4fa4')}
    metrics = {'W': aggregate(joined)}
    for tag, (path, digest) in baselines.items():
        data = a.rows(a.file(path, digest))
        require(len(data) == 18792, 'Baseline length differs')
        for x, y in zip(joined, data):
            a.check(all(x[k] == y[k] for k in FIELDS), tag+' exact paired full metadata '+str(x['window_index']))
            a.check({k:v['loss_tokens'] for k,v in x['query_history_bins'].items()} == {k:v['loss_tokens'] for k,v in y['query_history_bins'].items()}, tag+' same position targets')
        metrics[tag] = aggregate(data)
    for tag, mm in metrics.items():
        a.check(mm['total']['loss_tokens'] == 17418742 and len(mm['per_source']) == 6, tag+' complete targets/six sources')
        a.check(sum(v['loss_tokens'] for v in mm['position'].values()) == 17418742, tag+' position complete targets')
        a.check(abs(math.fsum(v['nll_sum'] for v in mm['per_source'].values())-mm['total']['nll_sum']) < 1e-6, tag+' source NLL sum')
        # Position diagnostics use per-token FP64 accumulation; reported row CE is FP32.
        a.check(abs(math.fsum(v['nll_sum'] for v in mm['position'].values())/17418742-mm['total']['nll']) < 1e-6, tag+' token-position NLL consistency')
    a.check(combined['status'] == 'complete_local_dev_union_audited' and combined['condition'] == 'W_fixed_local'
            and combined['single_uninterrupted_run'] is False
            and combined['prefix_status'] == 'evaluation_failed' and combined['prefix_error_type'] == 'TimeoutError',
            'wrapper union retains separate-run and original-timeout identity')
    a.check(combined['total']['windows'] == 18792 and combined['total']['loss_tokens'] == 17418742
            and abs(combined['total']['nll']-metrics['W']['total']['nll']) < 1e-12
            and combined['total']['attention_counts'] == dict(attention), 'independent full union equals wrapper summary')
    for name, values in metrics['W']['per_source'].items():
        a.check(combined['per_source'][name]['loss_tokens'] == values['loss_tokens']
                and abs(combined['per_source'][name]['nll']-values['nll']) < 1e-12, 'independent source union '+name)
    for name, values in metrics['W']['position'].items():
        item = combined['position_diagnostics']['query_history_bins'][name]
        a.check(item['loss_tokens'] == values['loss_tokens'] and abs(item['nll']-values['nll']) < 1e-12,
                'independent position union '+name)
    require(not a.report['failures'], 'Union checks failed; no full quality statistics published')
    differences = {tag: {'total': difference(metrics['W']['total'], mm['total']),
                         'per_source': {src: difference(values, mm['per_source'][src]) for src,values in metrics['W']['per_source'].items()},
                         'position': {pos: difference(values, mm['position'][pos]) for pos,values in metrics['W']['position'].items()}}
                   for tag,mm in metrics.items() if tag != 'W'}
    return {'status': 'passed_complete_two_segment_W_union_audit', 'complete_quality_evidence': True,
            'full_dev': metrics, 'W_minus_baseline': differences,
            'original_prefix_failure_preserved': {'status': ps['status'], 'error_type': ps['error_type'],
                'partial_metrics_only': ps['partial_metrics_only'], 'windows': 12402, 'raw_sha256': sha(pw)},
            'partition': {'prefix_indices': [0,12401], 'remainder_indices': [12402,18791], 'no_duplicates_or_omissions': True,
                          'complete_totals': EXPECTED, 'prefix_raw_sha256': sha(pw), 'remainder_raw_sha256': sha(rw),
                          'ordered_union_raw_sha256': hashlib.sha256(pw.read_bytes()+rw.read_bytes()).hexdigest(),
                          'original_files_never_rewritten': True},
            'physical_evaluation_counts': {'original_prefix': left_counts, 'new_remainder': right_counts,
                'union_forward_calls': 18792, 'union_backward_calls': 0, 'union_optimizer_updates': 0,
                'migration_engineering_replay_forward_calls_separate': 48,
                'migration_tiny_engineering_separate': {'model_forward_calls': 9, 'backward_calls': 7, 'updates': 3,
                                                      'component_forward_calls': 2, 'component_backward_calls': 2, 'support_calls': 28}},
            'timing_segments': {'original_prefix': {k:ps[k] for k in ('started_utc','completed_utc','elapsed_wall_seconds')},
                'new_remainder': {k:rs[k] for k in ('started_utc','completed_utc','elapsed_wall_seconds')},
                'new_authorized_launch': launch,
                'summed_scorer_active_wall_seconds': ps['elapsed_wall_seconds']+rs['elapsed_wall_seconds'],
                'interpretation': 'Two separate evaluator segments, excluding migration/setup/idle. Not a single uninterrupted timing measurement.'},
            'cost_boundary': 'Original timeout work and later remainder both count. No time or charge is erased. Original launch hourly price unknown; provider invoice/rate and idle intervals require separate accounting, never assumed zero.',
            'checkpoint_sha256': FINAL_SHA, 'checkpoint_training_protocol_sha256': TRAINING_PROTOCOL_SHA,
            'runtime': rs['runtime'], 'attention_counts': dict(attention),
            'remainder_receipt': receipt,
            'limitations': ['Evaluation resume only; original W training remains one completed1413-update run with no new optimizer steps.',
                            'This is the complete frozen development dataset, already used for development; not an independent unseen confirmation.',
                            'One paired seed/small architecture; no statistical equivalence, full Qwen architecture claim, or paper-level proof.',
                            'Reference full-score sparse code and hardware/migration differences do not establish speedup or lower end-to-end cost.',
                            'Original timeout and prefix are preserved rather than hidden by the successful union.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('metrics-audit.json'))
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.parent == Path(__file__).parent and not output.exists(), 'Fresh output required beside this audit script')
    a = Audit()
    a.report['script_sha256'] = sha(__file__)
    a.report['requested_manifest'] = str(args.manifest)
    a.report['requested_manifest_sha256'] = args.manifest_sha256
    try:
        path = args.manifest if args.manifest.is_absolute() else ROOT/args.manifest
        a.report.update(run(a, path, args.manifest_sha256))
    except BaseException as exc:
        a.report.update(status='audit_failed_no_full_quality_result', complete_quality_evidence=False,
                        error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    a.report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(a.report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': a.report['status'], 'checks': a.report['checks'], 'failures': a.report['failures'][:10],
                      'output': str(output), 'sha256': sha(output), 'error': a.report.get('error')}, ensure_ascii=False))
    return 0 if a.report['complete_quality_evidence'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
