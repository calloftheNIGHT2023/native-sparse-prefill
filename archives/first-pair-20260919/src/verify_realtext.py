"""Audit data hashes, complete per-run event chains, seed pairing and interventions."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_tree(folder):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    for row in manifest:
        assert sha(folder / row['path']) == row['sha256'], row['path']
    return len(manifest)


def verify_data(folder):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    records = manifest['examples']
    assert len(records) == sum(manifest['counts'].values())
    assert len({r['text_sha256'] for r in records}) == len(records)
    assert len({r['token_prefix_sha256'] for r in records}) == len(records)
    for row in records:
        assert sha(folder / row['trace_path']) == row['trace_sha256']
    downloads = json.loads((folder / 'download-manifest.json').read_text(encoding='utf-8'))
    project_root = folder.resolve().parents[1]
    for asset in downloads:
        assert sha(project_root / asset['path']) == asset['sha256'], asset['path']
    assert manifest['max_qk_attention_error'] <= 2e-5
    return {'folder': str(folder), 'examples_verified': len(records),
            'source_assets_hash_verified': len(downloads),
            'max_qk_attention_error': manifest['max_qk_attention_error'],
            'model_revision': manifest['model_revision']}


def verify_runs(folder):
    count = verify_tree(folder)
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    cfg = json.loads((folder / 'frozen-config.json').read_text(encoding='utf-8'))
    expected = len(cfg['seeds']) * len(cfg['methods']) * len(cfg['layer_ids'])
    assert summary['runs_completed'] == expected == len(summary['results'])
    groups, total_steps, clock_adjustments = defaultdict(list), 0, []
    for row in summary['results']:
        run = folder / f"layer{row['layer']}__seed{row['seed']}__{row['method']}"
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        assert events[0]['event'] == 'start' and events[-1]['event'] == 'complete'
        for a, b in zip(events, events[1:]):
            if a['utc'] > b['utc']:
                clock_adjustments.append({'run': run.name, 'before_utc': a['utc'], 'after_utc': b['utc'],
                    'before_step': a.get('step'), 'after_step': b.get('step'),
                    'note': 'Raw wall clock moved backward; preserved. Ordering uses step IDs and monotonic elapsed training time.'})
        steps = [e for e in events if e['event'] == 'optimizer_step']
        assert [s['step'] for s in steps] == list(range(1, row['updates'] + 1))
        assert all(a['reference_train_seconds'] <= b['reference_train_seconds'] for a, b in zip(steps, steps[1:]))
        total_steps += len(steps)
        assert all(s['candidate_token_pairs'] > 0 and s['reference_train_seconds'] >= 0 for s in steps)
        groups[(row['layer'], row['seed'])].append(run)
        if row['method'] == 'selected_budget_match':
            assert row['candidate_token_pairs_logical_only'] >= row['probe_logical_pair_budget']
    for folders in groups.values():
        first = torch.load(folders[0] / 'initial-indexer.pt', map_location='cpu', weights_only=True)
        for run in folders[1:]:
            other = torch.load(run / 'initial-indexer.pt', map_location='cpu', weights_only=True)
            assert all(torch.equal(first[k], other[k]) for k in first)
    return {'folder': str(folder), 'files_hash_verified': count, 'runs_verified': expected,
            'optimizer_events': total_steps, 'paired_initialization_groups': len(groups),
            'wall_clock_backward_events': clock_adjustments}


def verify_nll(folder):
    count = verify_tree(folder)
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    for row in summary['results']:
        assert len(row['per_example']) == summary['paragraphs_per_condition']
        if row['method'] == 'dense':
            assert row['max_dense_difference'] <= 1e-4
    return {'folder': str(folder), 'files_hash_verified': count, 'conditions': summary['conditions'],
            'paragraph_forward_passes': summary['conditions'] * summary['paragraphs_per_condition']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, nargs='*', default=[])
    parser.add_argument('--runs', type=Path, nargs='*', default=[])
    parser.add_argument('--nll', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior audit')
    started = datetime.now(timezone.utc).isoformat()
    report = {'started_utc': started, 'data': [verify_data(p) for p in args.data],
              'runs': [verify_runs(p) for p in args.runs], 'nll': [verify_nll(p) for p in args.nll],
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed'}
    if any(r['wall_clock_backward_events'] for r in report['runs']):
        report['status'] = 'data_and_step_audit_passed_with_recorded_wall_clock_adjustment'
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))
