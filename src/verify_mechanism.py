"""Audit the actual saved diagnostic runs, hashes, pairing and event completeness."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch


def main(root):
    start = datetime.now(timezone.utc).isoformat()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest:
        digest = hashlib.sha256((root / entry['path']).read_bytes()).hexdigest()
        if digest != entry['sha256']:
            raise AssertionError(f"Hash mismatch: {entry['path']}")
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    cfg = json.loads((root / 'frozen-config.json').read_text(encoding='utf-8'))
    assert summary['runs_completed'] == len(cfg['conditions']) * len(cfg['seeds']) * len(cfg['methods'])
    assert summary['started_utc'] < summary['finished_utc']
    grouped = defaultdict(list)
    updates = 0
    for result in summary['results']:
        folder = root / f"{result['condition']}__seed{result['seed']}__{result['method']}"
        events = [json.loads(line) for line in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        assert events[0]['event'] == 'start' and events[-1]['event'] == 'complete'
        assert all(a['utc'] <= b['utc'] for a, b in zip(events, events[1:]))
        steps = [e['step'] for e in events if e['event'] == 'optimizer_step']
        assert steps == list(range(1, cfg['updates'] + 1))
        updates += len(steps)
        grouped[(result['condition'], result['seed'])].append(folder)
    for folders in grouped.values():
        first = torch.load(folders[0] / 'initial-indexer.pt', map_location='cpu', weights_only=True)
        for folder in folders[1:]:
            other = torch.load(folder / 'initial-indexer.pt', map_location='cpu', weights_only=True)
            assert all(torch.equal(first[k], other[k]) for k in first)
    return {'started_utc': start, 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'passed', 'files_hash_verified': len(manifest),
            'completed_runs': summary['runs_completed'], 'optimizer_step_events': updates,
            'paired_initial_weights_verified_groups': len(grouped),
            'scope': 'indexer-only CPU synthetic runs; no language-model training or cloud GPU jobs'}


if __name__ == '__main__':
    root = Path(sys.argv[1])
    report = main(root)
    target = root.parent / f'{root.name}-verification.json'
    if target.exists():
        raise FileExistsError('Preserve prior verification; use an explicit new audit destination')
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))
