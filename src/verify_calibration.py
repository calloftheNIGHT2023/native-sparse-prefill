"""Audit the fixed factorial diagnostic and fresh-data exclusion evidence."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import torch

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / 'logs/indexer-calibration-v0-audit.json'
    if out.exists():
        raise FileExistsError('Preserve audit')
    started = datetime.now(timezone.utc).isoformat()
    root = ROOT / 'results/indexer-calibration-v0'
    manifest = read(root / 'manifest.json')
    for entry in manifest:
        assert sha(root / entry['path']) == entry['sha256']
    summary = read(root / 'summary.json')
    assert summary['runs_completed'] == len(summary['results']) == 72
    initials, grouped = defaultdict(list), defaultdict(list)
    steps_count, clocks = 0, []
    for row in summary['results']:
        folder = root / row['run_id']
        events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
        assert events[0]['event'] == 'start' and events[-1]['event'] == 'complete'
        assert all(b['monotonic_seconds'] >= a['monotonic_seconds'] for a,b in zip(events,events[1:]))
        clocks += [{'run_id': row['run_id'], 'before': a['utc'], 'after': b['utc']}
                   for a,b in zip(events,events[1:]) if b['utc'] < a['utc']]
        steps = [r for r in events if r['event']=='optimizer_step']
        assert [r['step'] for r in steps] == list(range(1,401))
        assert row['test_examples_loaded'] == 0
        assert sha(ROOT / row['case']['data'] / 'manifest.json') == row['trace_manifest_sha256']
        steps_count += len(steps)
        initial = torch.load(folder/'initial-indexer.pt', map_location='cpu', weights_only=True)
        if 'query_gain' in initial:
            assert (initial['query_gain']==0).all() and (initial['key_gain']==0).all()
        initials[(row['case']['data'],row['seed'])].append(initial)
        grouped[(row['case']['data'],row['variant'],row['method'])].append(row)
    for weights in initials.values():
        assert all(torch.equal(w[key],weights[0][key]) for w in weights for key in ['query.weight','key.weight'])
    old = read(ROOT/'data/realtext-v0-r1/manifest.json')
    new = read(ROOT/'data/realtext-calibrated-fresh-v0/manifest.json')
    for key in ['text_sha256','token_prefix_sha256']:
        assert not ({r[key] for r in old['examples']} & {r[key] for r in new['examples']})
        assert len({r[key] for r in new['examples']}) == 64
    source_dir = ROOT/'literature/indexer-fidelity-2026-09-13'
    sources = read(source_dir/'manifest.json')
    for source in sources:
        assert sha(source_dir/source['path']) == source['sha256']
    aggregate = [{'case': key[0], 'variant':key[1], 'method':key[2], 'seeds':len(rows),
        'mean_development': {k:statistics.mean(r['final_development'][k] for r in rows)
                             for k in rows[0]['final_development']}} for key,rows in grouped.items()]
    report = {'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed' if not clocks else 'passed_with_clock_warning', 'runs_verified':72,
        'files_hash_verified':len(manifest),'optimizer_steps_verified':steps_count,
        'paired_projection_groups_verified':len(initials),'source_files_verified':len(sources),
        'fresh_examples':64,'prior_examples':48,'text_and_prefix_overlap':0,
        'test_examples_loaded_in_factorial':0,'wall_clock_backward_events':clocks,
        'aggregate':aggregate}
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='aggregate'}))


if __name__ == '__main__':
    main()
