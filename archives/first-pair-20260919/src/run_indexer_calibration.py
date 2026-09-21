"""Fixed 2x2 calibration diagnostic; never loads the held-out test split."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback

import torch
from indexer_calibration import CalibratedIndexer
from sparse_reference import make_layout, select_blocks, sample_outside, subset_kl
from trace_math import prepare_trace, target_from_logits
from run_realtext import evaluate

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def health(idx, examples, layout, cfg):
    rows = []
    with torch.no_grad():
        for ex in examples:
            scores = idx(ex['hidden'], layout)
            visible = layout.visible_blocks[cfg['late_query_start']:]
            zero_queries = (((scores[cfg['late_query_start']:] > 0) & visible).sum(-1) == 0).float().mean().item()
            kl = subset_kl(scores, ex['full_target'], layout.visible_blocks).item()
            rows.append({'all_zero_query_fraction': zero_queries, 'full_teacher_kl': kl})
    return {**evaluate(idx, examples, layout, cfg),
            **{k: statistics.mean(row[k] for row in rows) for k in rows[0]}}


def main(args):
    cfg = json.loads(args.config.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'frozen-config.json', cfg)
    sources = out / 'source-snapshot'
    sources.mkdir()
    for name in ['run_indexer_calibration.py', 'indexer_calibration.py', 'sparse_reference.py', 'trace_math.py', 'run_realtext.py']:
        shutil.copy2(ROOT / 'src' / name, sources / name)
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    started, global_timer = utc(), time.perf_counter()
    save(out / 'environment.json', {'started_utc': started, 'device': 'cpu', 'python': sys.version, 'torch': torch.__version__})
    results = []
    for case in cfg['cases']:
        data = ROOT / case['data']
        trace_cfg = json.loads((data / 'frozen-config.json').read_text())
        manifest = json.loads((data / 'manifest.json').read_text())
        layout = make_layout([0] * trace_cfg['sequence_length'], trace_cfg['block_size'])
        examples = {'train': [], 'validation': []}
        for record in manifest['examples']:
            if record['split'] not in examples:
                continue
            path = data / record['trace_path']
            assert sha(path) == record['trace_sha256']
            ex = torch.load(path, map_location='cpu', weights_only=True)
            examples[record['split']].append(prepare_trace(ex['layers'][str(case['layer'])], layout))
        for seed in cfg['seeds']:
            for variant, factors in cfg['variants'].items():
                for method in cfg['methods']:
                    name = f"{data.name}__layer{case['layer']}__seed{seed}__{variant}__{method}"
                    folder = out / name
                    folder.mkdir()
                    timer, run_start = time.perf_counter(), utc()
                    def event(kind, **extra):
                        with (folder / 'events.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps({'utc': utc(), 'monotonic_seconds': time.perf_counter() - timer,
                                'event': kind, **extra}) + '\n')
                    event('start', seed=seed, variant=variant, method=method)
                    try:
                        torch.manual_seed(seed)
                        idx = CalibratedIndexer(examples['train'][0]['hidden'].shape[-1], trace_cfg['index_heads'],
                            trace_cfg['index_head_dim'], trace_cfg['rotary_dim'], **factors)
                        torch.save(idx.state_dict(), folder / 'initial-indexer.pt')
                        generator = torch.Generator().manual_seed(seed + 8888)
                        optimizer = torch.optim.AdamW(idx.parameters(), lr=cfg['learning_rate'], weight_decay=0)
                        initial = health(idx, examples['validation'], layout, trace_cfg)
                        event('development_eval', step=0, **initial)
                        for step in range(cfg['updates']):
                            ex = examples['train'][step % len(examples['train'])]
                            scores = idx(ex['hidden'], layout)
                            selected = select_blocks(scores, layout.visible_blocks, trace_cfg['selected_blocks'])
                            if method == 'full_supervision':
                                support = layout.visible_blocks
                            elif method == 'outside_probe':
                                support = selected | sample_outside(selected, layout.visible_blocks, trace_cfg['probe_blocks'], generator)
                            else:
                                support = selected
                            loss = subset_kl(scores, target_from_logits(ex['logits'], support, layout), support)
                            assert torch.isfinite(loss)
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            grad = torch.nn.utils.clip_grad_norm_(idx.parameters(), cfg['gradient_clip_norm'], error_if_nonfinite=True)
                            optimizer.step()
                            event('optimizer_step', step=step + 1, loss=loss.item(), gradient_norm=grad.item())
                            if (step + 1) % cfg['evaluation_every'] == 0:
                                final = health(idx, examples['validation'], layout, trace_cfg)
                                event('development_eval', step=step + 1, **final)
                        torch.save(idx.state_dict(), folder / 'final-indexer.pt')
                        result = {'run_id': name, 'case': case, 'seed': seed, 'variant': variant, 'method': method,
                            'started_utc': run_start, 'finished_utc': utc(), 'elapsed_seconds': time.perf_counter() - timer,
                            'updates': cfg['updates'], 'initial_development': initial, 'final_development': final,
                            'trace_manifest_sha256': sha(data / 'manifest.json'), 'test_examples_loaded': 0}
                        save(folder / 'result.json', result)
                        event('complete')
                        results.append(result)
                        print(json.dumps({'run': name, **final}), flush=True)
                    except Exception:
                        event('failed', traceback=traceback.format_exc())
                        raise
    summary = {'started_utc': started, 'finished_utc': utc(), 'elapsed_seconds': time.perf_counter()-global_timer,
               'runs_completed': len(results), 'results': results, 'test_examples_loaded': 0,
               'note': cfg['purpose']}
    save(out / 'summary.json', summary)
    save(out / 'manifest.json', [{'path': p.relative_to(out).as_posix(), 'sha256': sha(p)}
                               for p in out.rglob('*') if p.is_file()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
