"""Verify the published first-pair evidence using the standard library only.

No model evaluation, downloads or training. Run inside the GitHub snapshot.
"""
import gzip
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'EXPORT_MANIFEST.json').read_bytes())
    for entry in manifest['files']:
        raw = (root / entry['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry['export_sha256'], entry['path']
        if entry.get('decompressed_sha256'):
            assert hashlib.sha256(gzip.decompress(raw)).hexdigest() == entry['decompressed_sha256']
    runs = {}
    for mode in ('dense', 'sparse'):
        folder = root / 'evidence/babylm-first-pair' / mode
        rows = [json.loads(line) for line in gzip.decompress(
            (folder / 'events.redacted.jsonl.gz').read_bytes()).splitlines()]
        summary = json.loads((folder / 'summary.json').read_bytes())
        updates = [r for r in rows if r['type'] == 'update']
        stops = [r for r in rows if r['type'] == 'run_stop']
        evals = [r for r in rows if r['type'] == 'evaluation']
        assert summary['status'] == 'word_budget_reached'
        assert summary['counts']['scientific_updates'] == len(updates) == 14122
        assert summary['counts']['input_tokens'] == 163224857
        assert summary['counts']['word_exposures'] == 99998882
        assert summary['counts'] == updates[-1]['counts'] == stops[-1]['counts']
        assert summary['eval_counts'] == evals[-1]['eval_counts']
        assert len(evals) == summary['eval_counts']['evaluations'] == 58
        assert [r['event_id'] for r in rows] == list(range(1, len(rows) + 1))
        for source, expected in summary['source_hashes'].items():
            if '/site-packages/' in source:
                continue  # External dependency version/hash is recorded, not vendored.
            marker = '/babylm-de-20260918-v0/'
            assert marker in source
            file = root / source.split(marker, 1)[1]
            assert hashlib.sha256(file.read_bytes()).hexdigest() == expected, str(file)
        runs[mode] = (summary, updates, evals)
    dense, sparse = runs['dense'], runs['sparse']
    assert dense[0]['data_fingerprint'] == sparse[0]['data_fingerprint']
    assert dense[0]['initial_parameter_hashes'] == {
        k: v for k, v in sparse[0]['initial_parameter_hashes'].items() if '.indexer.' not in k}
    for d, s in zip(dense[1], sparse[1]):
        assert d['counts'] == s['counts'] and d['window_ids'] == s['window_ids']
        assert d['lr']['backbone'] == s['lr']['backbone']
    for d, s in zip(dense[2], sparse[2]):
        assert d['counts'] == s['counts']
        assert d['metrics']['window_indices'] == s['metrics']['window_indices']
        assert d['metrics']['manifest_sha256'] == s['metrics']['manifest_sha256']
    print(json.dumps({'status': 'passed', 'files_verified': len(manifest['files']),
                      'paired_updates_verified': len(dense[1]),
                      'paired_panel_evaluations_verified': len(dense[2]),
                      'dense_final_panel_nll': dense[2][-1]['metrics']['total']['nll'],
                      'sparse_final_panel_nll': sparse[2][-1]['metrics']['total']['nll'],
                      'scientific_source_hashes_match_original_runs': True,
                      'model_calls': 0,
                      'scope': 'Archive/provenance checks only; no full-dev, numerical replay, quality equivalence or speed claim'}, indent=2))


if __name__ == '__main__':
    main()
