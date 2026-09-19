"""Inspect saved indexers and logs without optimizer updates or test-set tuning."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import torch
from sparse_reference import QSAIndexer, make_layout
from indexer_calibration import make_indexer


def main(args):
    if args.output.exists():
        raise FileExistsError('Preserve previous health audits')
    torch.set_num_threads(4)
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for root in args.runs:
        cfg = json.loads((root / 'frozen-config.json').read_text())
        env = json.loads((root / 'environment.json').read_text())
        data = Path(env['data_path'])
        manifest = json.loads((data / 'manifest.json').read_text())
        layout = make_layout([0] * cfg['sequence_length'], cfg['block_size'])
        examples = [torch.load(data / r['trace_path'], map_location='cpu', weights_only=True)
                    for r in manifest['examples'] if r['split'] == 'validation']
        summary = json.loads((root / 'summary.json').read_text())
        for row in summary['results']:
            run = root / f"layer{row['layer']}__seed{row['seed']}__{row['method']}"
            states = torch.load(run / 'final-indexer.pt', map_location='cpu', weights_only=True)
            idx = make_indexer(states['key.weight'].shape[1], cfg)
            idx.load_state_dict(states)
            zero_pairs, total_pairs, dead_queries, total_queries = 0, 0, 0, 0
            late = torch.arange(cfg['sequence_length']) >= cfg['late_query_start']
            with torch.no_grad():
                for ex in examples:
                    scores = idx(ex['layers'][str(row['layer'])]['hidden'], layout)
                    visible = layout.visible_blocks & late[:, None]
                    zero_pairs += ((scores == 0) & visible).sum().item()
                    total_pairs += visible.sum().item()
                    dead_queries += (((scores > 0) & visible).sum(-1)[late] == 0).sum().item()
                    total_queries += late.sum().item()
            steps = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()
                     if '"optimizer_step"' in line]
            last = steps[-min(100, len(steps)):]
            rows.append({'run_group': root.name, 'layer': row['layer'], 'seed': row['seed'],
                'method': row['method'], 'late_validation_visible_score_zero_fraction': zero_pairs / total_pairs,
                'late_validation_all_zero_query_fraction': dead_queries / total_queries,
                'last_100_steps_zero_gradient_fraction': statistics.mean(s['gradient_norm'] == 0 for s in last),
                'last_100_steps_mean_gradient_norm': statistics.mean(s['gradient_norm'] for s in last)})
    report = {'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
              'runs_inspected': len(rows), 'new_optimizer_updates': 0, 'rows': rows,
              'scope': 'Read-only final score audit on validation traces and saved training gradients. No hyperparameter selection.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'runs_inspected': len(rows), 'new_optimizer_updates': 0,
                      'fully_dead_runs': [r for r in rows if r['late_validation_all_zero_query_fraction'] == 1]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
