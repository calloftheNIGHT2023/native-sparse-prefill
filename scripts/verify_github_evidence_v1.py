"""Offline raw-evidence verification of current two-seed D/W and historical pair.

Standard library only. No tensor loading, model calls, data downloads or network.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'results/babylm-dw-seed-confirmation-20260921-v1/'
TARGETS, INPUTS, WINDOWS = 17418742, 17437534, 18792
SECRET = re.compile(r'(?<![A-Za-z0-9_-])(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|'
                    r'AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{24,}|rpa_[A-Za-z0-9_-]{20,})|'
                    r'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----')
RELAY = re.compile(r'[A-Za-z0-9_-]+@ssh\.runpod\.io')
KEYPATH = re.compile(r'(?:[A-Za-z]:[\\/]|~[\\/]|/)[^\s\"\'<>]*[\\/]\.ssh[\\/][^\s\"\'<>]+')


def require(value, message):
    if not value: raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def raw(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), 'Evidence path escapes snapshot')
    data = path.read_bytes()
    return gzip.decompress(data) if str(relative).endswith('.gz') else data


def load(relative):
    return json.loads(raw(relative))


def records(relative):
    data = raw(relative)
    require(data.endswith(b'\n'), 'Incomplete JSONL: ' + relative)
    return [json.loads(line) for line in data.splitlines()]


def close(left, right, label, tolerance=1e-10):
    require(math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance, label)


def metrics(rows):
    require([r['window_index'] for r in rows] == list(range(WINDOWS)), 'Dev windows duplicated/missing/reordered')
    require(sum(r['loss_tokens'] for r in rows) == TARGETS and sum(r['input_tokens'] for r in rows) == INPUTS,
            'Full dev denominator differs')
    sums, counts = defaultdict(list), defaultdict(int)
    for row in rows:
        n = row['loss_tokens']
        require(row['forward_calls'] == 1 and row['grad_enabled'] is False and row['model_training'] is False, 'Dev work contract differs')
        require(n == max(row['input_tokens'] - 1, 0), 'Next-token count differs')
        if n:
            close(row['nll'] * n, row['nll_sum'], 'Row NLL sum differs', 1e-6)
            close(math.exp(row['nll']), row['ppl'], 'Row PPL differs', 1e-7)
        else:
            require(row['nll_sum'] == 0 and row['nll'] is None and row['ppl'] is None, 'Zero-target row differs')
        for key in ('all', row['source']):
            sums[key].append(row['nll_sum']); counts[key] += n
    return {key: {'nll': math.fsum(values) / counts[key], 'ppl': math.exp(math.fsum(values) / counts[key]),
                  'loss_tokens': counts[key], 'nll_sum': math.fsum(values)} for key, values in sums.items()}


def compare_metadata(left, right):
    keys = ('window_index', 'source_index', 'segment_index_in_source', 'source_token_start', 'source_token_end',
            'word_exposures', 'input_tokens', 'loss_tokens', 'source')
    for a, b in zip(left, right):
        require(all(a[k] == b[k] for k in keys), 'Paired dev data differs')
        require(set(a['query_history_bins']) == set(b['query_history_bins']), 'Position bins differ')
        for key in a['query_history_bins']:
            require(a['query_history_bins'][key]['loss_tokens'] == b['query_history_bins'][key]['loss_tokens'], 'Position denominator differs')


def audit_training(index, arm, pinned):
    base = PREFIX + arm + '/train/run/'
    summary = load(index['terminal'][base + 'summary.json'])
    protocol = load(index['terminal'][base + 'protocol.json'])
    events = records(index['terminal'][base + 'events.jsonl'])
    updates = [e for e in events if e['type'] == 'update']
    require([e['event_id'] for e in events] == list(range(1, len(events) + 1)), 'Noncontiguous training events')
    require(summary['status'] == 'epoch_complete' and summary['scope'] == protocol['scope'] == 'scientific', 'Incomplete science training')
    require(protocol['backbone_seed'] == 20260921 and protocol['data_order_seed'] == 20260919, 'Frozen seeds differ')
    require(protocol['max_epochs'] == 1 and protocol['scientific_condition'] == ('D_dense' if arm == 'D' else 'W_fixed_local'), 'Condition/epoch differs')
    require(summary['counts']['updates'] == summary['counts']['scientific_updates'] == len(updates) == 1413
            and summary['counts']['engineering_updates'] == 0, 'Scientific updates differ')
    require(summary['counts']['forward_calls'] == summary['counts']['backward_calls'] == 22598
            and summary['counts']['input_tokens'] == 16325414
            and summary['counts']['loss_tokens'] == 16302816
            and summary['counts']['word_exposures'] == 10001709, 'Science token/work counts differ')
    require(not any(e['type'] in {'failure', 'resume', 'run_resume', 'consumed_without_update'} for e in events), 'Unexpected scientific trajectory branch')
    require([e['counts']['updates'] for e in updates] == list(range(1, 1414)), 'Update sequence differs')
    ids = []
    for update in updates:
        require(update['aux_weight_applied'] == 0 and update['loss_token_weighted_aux'] == 0, 'Unexpected routing auxiliary objective')
        require(len(update['windows']) == (6 if update['counts']['updates'] == 1413 else 16), 'Microbatch size differs')
        for window in update['windows']:
            require(window['epoch'] == 0 and all(window[k] for k in ('forward_started', 'forward_completed', 'backward_started', 'backward_completed')), 'Incomplete window work')
            ids.append(window['window_index'])
    require(sorted(ids) == list(range(22598)), 'Training coverage differs')
    evals = [e for e in events if e['type'] == 'evaluation']
    require([e['counts']['updates'] for e in evals] == [0, 250, 500, 750, 1000, 1250, 1413]
            and summary['eval_counts']['forward_calls'] == 336, 'Monitoring work differs')
    for name, expected in pinned.items():
        require(digest((ROOT / name).read_bytes()) == expected, 'Current source changed: ' + name)
        matches = [d for p, d in summary['source_hashes'].items() if p.replace('\\', '/').endswith('/' + name)]
        require(matches and all(d == expected for d in matches), 'Run/source pin mismatch: ' + name)
    return summary, protocol, updates


def audit_cost(index, export_entries):
    if 'cost' not in index:
        return None
    base = 'results/babylm-dw-cost-profile-20260921-v0/'
    summary = load(index['cost'][base + 'summary.json'])
    events = records(index['cost'][base + 'events.jsonl'])
    original = load(index['cost_manifest'])
    require(len(original['files']) == len(index['cost']) == 6, 'Cost allowlist differs')
    for item in original['files']:
        exported = export_entries[index['cost'][item['path']]]
        require(exported['source_sha256'] == item['sha256'] and exported['source_bytes'] == item['bytes'], 'Cost original hash mapping differs')
    require(summary['status'] == 'complete_read_only_cost_diagnostic' and summary['scope'] == 'engineering_cost_diagnostic'
            and summary['no_optimizer_instantiated'] is True, 'Cost diagnostic incomplete/mislabelled')
    expected_types = {'forward_attempt': 36, 'forward_complete': 36, 'backward_attempt': 32, 'backward_complete': 32,
                      'warmup_complete': 4, 'hook_equivalence': 16, 'paired_measurement_complete': 16, 'arm_complete': 2}
    require(dict(Counter(e['type'] for e in events)) == expected_types, 'Cost raw call counts differ')
    require(events[-1]['counts'] == summary['counts'], 'Cost terminal counters differ')
    for key, n in [('forward_attempts', 36), ('forward_calls', 36), ('backward_attempts', 32), ('backward_calls', 32),
                   ('warmup_forward_calls', 4), ('optimizer_updates', 0), ('scientific_updates', 0)]:
        require(summary['counts'][key] == n, 'Cost work field differs: ' + key)
    for source, expected in summary['source_hashes'].items():
        require(digest((ROOT / source).read_bytes()) == expected, 'Cost source hash differs: ' + source)
    for event in events:
        require(event['counts']['optimizer_updates'] == event['counts']['scientific_updates'] == 0, 'Cost changed scientific counters')
        if event['type'] == 'hook_equivalence':
            values = [event['comparison']['loss'], *event['comparison']['gradients'].values()]
            require(len(values) == 147 and all(v['passed'] is True and v['max_tolerance_ratio'] <= 1 for v in values), 'Hook equivalence failed')
    shares = {}
    for arm, result in summary['arms'].items():
        rows = result['rows']
        require(len(rows) == 8 and len({r['window_index'] for r in rows}) == 8, 'Cost sample duplicated')
        require(result['model_state_before_sha256'] == result['model_state_after_sha256']
                and result['weights_unchanged'] is True and result['checkpoint_file_unchanged'] is True, 'Cost changed weights')
        recorded = [e['row'] for e in events if e['type'] == 'paired_measurement_complete' and e['arm'] == arm]
        require(recorded == rows, 'Cost summary/raw measurements differ')
        group_sums = defaultdict(list)
        for row in rows:
            require(row['equivalence_passed'] is True, 'Cost row equivalence failed')
            groups = defaultdict(list)
            for component in row['instrumented']['forward_components']:
                require(math.isfinite(component['stream_elapsed_ms']) and component['stream_elapsed_ms'] >= 0, 'Invalid component time')
                groups[component['group']].append(component['stream_elapsed_ms'])
            for group, values in groups.items():
                total = math.fsum(values)
                close(total, row['forward_group_stream_ms'][group], 'Component sum differs', 1e-8)
                close(total / row['instrumented']['forward_stream_ms'], row['forward_group_share'][group], 'Component share differs')
                group_sums[group].append(total)
        denominator = math.fsum(r['instrumented']['forward_stream_ms'] for r in rows)
        shares[arm] = {}
        for group, values in group_sums.items():
            share = math.fsum(values) / denominator
            close(share, result['pooled_instrumented_forward_share'][group], 'Pooled forward share differs')
            shares[arm][group] = share
        for mode in ('bare', 'instrumented'):
            for metric in ('forward_stream_ms', 'backward_stream_ms', 'forward_wall_seconds', 'backward_wall_seconds'):
                values = [r[mode][metric] for r in rows]; reported = result['timing_summary'][mode + '.' + metric]
                close(math.fsum(values), reported['sum'], 'Cost timing sum differs', 1e-8)
                close(statistics.median(values), reported['median'], 'Cost timing median differs', 1e-8)
    return {'status': 'raw_cost_counters_and_forward_shares_recomputed', 'engineering_forward_calls': 36,
            'engineering_backward_calls': 32, 'optimizer_updates': 0, 'pooled_instrumented_forward_share': shares,
            'elapsed_wall_seconds': summary['elapsed_wall_seconds'],
            'scope': 'Eight windows on shared GPU, forward-only component proportions; not training F+B or end-to-end speed evidence.'}


def main():
    manifest = load('EXPORT_MANIFEST.json'); index = load('EVIDENCE_INDEX.json')
    require(manifest['schema_version'] == 2 and manifest['selected_terminal_artifacts'] == 45, 'Wrong snapshot format')
    listed = set(); export_entries = {}
    for entry in manifest['files']:
        require(entry['path'] not in listed, 'Duplicate export entry'); listed.add(entry['path'])
        export_entries[entry['path']] = entry
        payload = (ROOT / entry['path']).read_bytes()
        require(digest(payload) == entry['export_sha256'] and len(payload) == entry['export_bytes'], 'Export hash/size differs: ' + entry['path'])
        plain = gzip.decompress(payload) if entry.get('decompressed_sha256') else payload
        if entry.get('decompressed_sha256'): require(digest(plain) == entry['decompressed_sha256'], 'Gzip content differs')
        if entry['byte_preserved'] and not entry['path'].startswith(index['historical_root'] + '/'):
            require(digest(plain) == entry['source_sha256'], 'Preserved original bytes differ')
        text = plain.decode('utf-8-sig')
        require(not SECRET.search(text) and not RELAY.search(text) and not KEYPATH.search(text), 'Credential/connection signature: ' + entry['path'])
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
              and '.git' not in p.parts and '__pycache__' not in p.parts}
    require(actual == listed | {'EXPORT_MANIFEST.json'}, 'Unmanifested or missing files')
    terminal_manifest = load('evidence/seed-20260921/terminal-manifest.redacted.json')
    require(len(terminal_manifest['receipts']) == len(index['terminal']) == 45, 'Terminal evidence count differs')
    for item in terminal_manifest['receipts']:
        exported = export_entries[index['terminal'][item['remote_relative_path']]]
        require(exported['source_sha256'] == item['sha256'] and exported['source_bytes'] == item['size_bytes'], 'Terminal original/export mapping differs')
    old = ROOT / index['historical_root'] / 'scripts/verify_github_evidence_v0.py'
    historical = subprocess.run([sys.executable, str(old)], cwd=old.parents[1], capture_output=True, text=True, timeout=90)
    require(historical.returncode == 0, 'Unchanged historical verifier failed: ' + historical.stderr[-1000:])
    historical_report = json.loads(historical.stdout)
    pinned = manifest['required_current_scientific_source_sha256']
    ds, dp, du = audit_training(index, 'D', pinned); ws, wp, wu = audit_training(index, 'W', pinned)
    require(ds['initial_parameter_hashes'] == ws['initial_parameter_hashes'], 'New seed pair starts from different backbone')
    require(ds['data_fingerprint'] == ws['data_fingerprint'], 'New seed training data differs')
    for d, w in zip(du, wu):
        require(d['windows'] == w['windows'] and d['counts'] == w['counts'] and d['lr'] == w['lr'], 'Paired training data/count/LR differs')
    current_report = load(index['reports']['babylm-dw-seed-full-dev-audit-20260921/metrics-audit.json'])
    first_report = load(index['reports']['babylm-stage-w-complete-20260921/metrics-audit.json'])
    pairs = []
    previous_rows = None
    for seed in (20260917, 20260921):
        if seed == 20260917:
            drows = records(index['seed1_raw']['D'])
            prefix, remainder = records(index['seed1_raw']['W_prefix']), records(index['seed1_raw']['W_remainder'])
            require([r['window_index'] for r in prefix] == list(range(12402))
                    and [r['window_index'] for r in remainder] == list(range(12402, WINDOWS)), 'W historical split differs')
            wrows = prefix + remainder; report = first_report
        else:
            drows = records(index['terminal'][PREFIX + 'D/eval/windows.jsonl'])
            wrows = records(index['terminal'][PREFIX + 'W/eval/windows.jsonl']); report = current_report
        compare_metadata(drows, wrows)
        if previous_rows is not None: compare_metadata(previous_rows, drows)
        previous_rows = drows
        result = {}
        for arm, rows in (('D', drows), ('W', wrows)):
            result[arm] = metrics(rows)
            close(result[arm]['all']['nll'], report['full_dev'][arm]['total']['nll'], 'Reported NLL differs')
            close(result[arm]['all']['ppl'], report['full_dev'][arm]['total']['ppl'], 'Reported PPL differs')
            for source, value in result[arm].items():
                if source != 'all': close(value['nll'], report['full_dev'][arm]['per_source'][source]['nll'], 'Source NLL differs')
        pairs.append({'backbone_seed': seed, 'D': result['D']['all'], 'W': result['W']['all'],
                      'W_minus_D_NLL': result['W']['all']['nll'] - result['D']['all']['nll']})
    reported = current_report['two_seed_descriptive']['paired_results']
    for derived, expected in zip(pairs, reported):
        require(derived['backbone_seed'] == expected['backbone_seed'], 'Seed order changed')
        close(derived['W_minus_D_NLL'], expected['W_minus_D']['nll_difference'], 'Two-seed delta differs')
    execution = load(index['reports']['babylm-dw-seed-full-dev-audit-20260921/execution-audit.json'])
    require(execution['status'] == 'passed_terminal_execution_cost_audit' and execution['checks_failed'] == 0, 'Execution audit failed')
    cost_report = audit_cost(index, export_entries)
    print(json.dumps({'status': 'passed', 'manifest_files_verified': len(listed),
                      'historical_verifier': historical_report, 'current_terminal_artifacts': len(index['terminal']),
                      'new_seed_training_updates_verified': 2826, 'new_seed_training_forward_calls': 45196,
                      'new_seed_panel_forward_calls': 672, 'new_seed_full_dev_forward_calls': 37584,
                      'two_seed_paired_results_recomputed_from_raw_rows': pairs,
                      'cost_diagnostic': cost_report,
                      'model_calls': 0, 'weight_files_loaded': 0, 'secret_signature_matches': 0,
                      'scope': 'Offline provenance/count/metric recomputation, not tensor replay or complete reproducibility; no significance, novelty or speed claim.'}, indent=2))


if __name__ == '__main__':
    main()
