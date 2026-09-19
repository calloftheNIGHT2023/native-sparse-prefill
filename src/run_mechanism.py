"""Run frozen-plan, CPU-only QSA-indexer diagnostics with complete event logs.

The teacher is supplied synthetic Q/K, not a learned language model. Every method
sees the same training inputs; seed repeats are independent indexer initializations
and data draws. The shifted teacher changes its query subspace halfway through.
Dense diagnostic tensors are deliberately used: logical score counts are NOT FLOPs
or actual sparse-kernel runtimes. Never interpret results as a Qwen benchmark.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time
import traceback

import torch
from sparse_reference import (QSAIndexer, make_layout, select_blocks, sample_outside,
                              expand_blocks, teacher_block_distribution, subset_kl)

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace(seed, n, r):
    gen = torch.Generator().manual_seed(seed)
    keys = torch.randn(n // r, 8, generator=gen)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    targets = []
    for i in range(n):
        # Once possible, point to a completed nonlocal block at least one block back.
        upper = max(1, i // r - 1)
        targets.append(int(torch.randint(upper, (1,), generator=gen)))
    query = keys[torch.tensor(targets)] + 0.1 * torch.randn(n, 8, generator=gen)
    content = keys.repeat_interleave(r, 0) + 0.03 * torch.randn(n, 8, generator=gen)
    return torch.cat((query, content), dim=-1), torch.tensor(targets)


def teacher(hidden, shifted):
    # Controlled nonstationarity, not an empirical backbone-training trajectory.
    query = hidden[:, :8]
    if shifted:
        query = query.roll(4, dims=-1)
    key = hidden[:, 8:]
    weights = hidden.new_tensor([1., .6, 1.4, .8, 1.2, .7, 1.3, .9])
    q = torch.stack((query * 4, query * weights * 4), dim=1)
    k = torch.stack((key * 4, key * weights * 4), dim=1)
    return q, k


def metrics(indexer, examples, layout, cfg, shifted):
    rows = []
    with torch.no_grad():
        for x, targets in examples:
            selected = select_blocks(indexer(x, layout), layout.visible_blocks, cfg['selected_blocks'])
            q, k = teacher(x, shifted)
            full = teacher_block_distribution(q, k, layout.visible_blocks, layout)
            late = torch.arange(len(x)) >= int(len(x) * cfg['late_query_fraction'])
            mass = (full * selected).sum(-1)[late].mean().item()
            strongest = full.argmax(-1)
            recall = selected.gather(1, strongest[:, None]).squeeze(1)[late].float().mean().item()
            rows.append({'teacher_mass': mass, 'teacher_top1_recall': recall})
            if not shifted:
                # Content target recall is meaningful only before the synthetic teacher shift.
                rows[-1]['constructed_target_recall'] = selected.gather(1, targets[:, None]).squeeze(1)[late].float().mean().item()
    return {key: statistics.mean(row[key] for row in rows) for key in rows[0]}


def run_one(cfg, seed, condition, method, folder):
    started, timer = utc(), time.perf_counter()
    folder.mkdir()
    event_file = folder / 'events.jsonl'
    def event(kind, **fields):
        with event_file.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'utc': utc(), 'event': kind, **fields}) + '\n')
    event('start', seed=seed, method=method, condition=condition)
    try:
        torch.manual_seed(seed)
        idx = QSAIndexer(cfg['model_dim'], cfg['index_heads'], cfg['index_head_dim'], cfg['rotary_dim'])
        initial_path = folder / 'initial-indexer.pt'
        torch.save(idx.state_dict(), initial_path)
        n, r = cfg['tokens_per_sequence'], cfg['block_size']
        layout = make_layout([0] * n, r)
        train = [trace(seed * 100000 + j, n, r) for j in range(cfg['train_sequences'])]
        dev = [trace(seed * 100000 + 1000 + j, n, r) for j in range(cfg['development_sequences'])]
        # Final held-out traces are never used for gradients or checkpoint selection.
        tests = [trace(seed * 100000 + 2000 + j, n, r) for j in range(cfg['test_sequences'])]
        probe_gen = torch.Generator().manual_seed(seed + 8765)
        optim = torch.optim.AdamW(idx.parameters(), lr=cfg['learning_rate'], weight_decay=0)
        curve, logical_pairs, train_seconds = [], 0, 0.0
        initial_metrics = metrics(idx, dev, layout, cfg, False)
        curve.append({'step': 0, 'teacher_shifted': False, **initial_metrics})
        event('development_eval', **curve[-1])
        for step in range(cfg['updates']):
            start_step = time.perf_counter()
            shifted = condition == 'teacher_shift_halfway' and step >= cfg['updates'] // 2
            x, _ = train[step % len(train)]
            q, k = teacher(x, shifted)
            scores = idx(x, layout)
            selected = select_blocks(scores, layout.visible_blocks, cfg['selected_blocks'])
            if method == 'full_supervision' or (method == 'dense_warmup' and step < cfg['warmup_updates']):
                support = layout.visible_blocks
            elif method == 'outside_probe':
                support = selected | sample_outside(selected, layout.visible_blocks, cfg['probe_blocks'], probe_gen)
            else:
                support = selected
            target = teacher_block_distribution(q, k, support, layout)
            loss = subset_kl(scores, target, support)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            logical_pairs += int(expand_blocks(support, layout).sum())
            train_seconds += time.perf_counter() - start_step
            event('optimizer_step', step=step + 1, loss=loss.item(), teacher_shifted=shifted,
                  cumulative_reference_train_seconds=train_seconds,
                  cumulative_candidate_token_pairs=logical_pairs)
            if (step + 1) % cfg['evaluation_every'] == 0:
                row = {'step': step + 1, 'teacher_shifted': shifted,
                       'cumulative_candidate_token_pairs': logical_pairs,
                       'cumulative_reference_train_seconds': train_seconds,
                       **metrics(idx, dev, layout, cfg, shifted)}
                curve.append(row)
                event('development_eval', **row)
        final = metrics(idx, tests, layout, cfg, condition == 'teacher_shift_halfway')
        torch.save(idx.state_dict(), folder / 'final-indexer.pt')
        save(folder / 'development-curve.json', curve)
        result = {'seed': seed, 'condition': condition, 'method': method,
                  'started_utc': started, 'finished_utc': utc(),
                  'elapsed_seconds_including_evaluation': time.perf_counter() - timer,
                  'reference_train_seconds': train_seconds, 'updates': cfg['updates'],
                  'candidate_token_pairs_logical_only': logical_pairs,
                  'actual_reference_dense_qk_pairs': n * n * cfg['updates'] * 2,
                  'held_out': final, 'initial_development': initial_metrics,
                  'initial_weights_sha256': hash_file(initial_path), 'status': 'complete',
                  'research_scope': 'synthetic indexer-only diagnostic'}
        save(folder / 'result.json', result)
        event('complete', result='result.json')
        return result
    except Exception:
        event('failed', traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/mechanism-v0.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if cfg['device'] != 'cpu':
        raise ValueError('This runner is CPU-only')
    if args.output.exists():
        raise FileExistsError('Preserve previous runs; choose a new output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    started = utc()
    save(args.output / 'frozen-config.json', cfg)
    source_dir = args.output / 'source-snapshot'
    source_dir.mkdir()
    for path in [ROOT / 'src/sparse_reference.py', Path(__file__), ROOT / 'tests/test_sparse_reference.py']:
        shutil.copy2(path, source_dir / path.name)
    save(args.output / 'environment.json', {'python': sys.version, 'torch': torch.__version__,
                                          'platform': platform.platform(), 'device': 'cpu',
                                          'started_utc': started, 'config_sha256': hash_file(args.config)})
    results = []
    for condition in cfg['conditions']:
        for seed in cfg['seeds']:
            for method in cfg['methods']:
                name = f'{condition}__seed{seed}__{method}'
                result = run_one(cfg, seed, condition, method, args.output / name)
                results.append(result)
                print(json.dumps({'run': name, **result['held_out']}), flush=True)
    grouped = defaultdict(list)
    for row in results:
        grouped[(row['condition'], row['method'])].append(row)
    summary = []
    for (condition, method), group in grouped.items():
        metric_names = group[0]['held_out'].keys()
        summary.append({'condition': condition, 'method': method, 'seeds': len(group),
                        'held_out_mean': {key: statistics.mean(row['held_out'][key] for row in group) for key in metric_names},
                        'held_out_sample_std': {key: statistics.stdev(row['held_out'][key] for row in group) for key in metric_names},
                        'logical_candidate_pairs_mean': statistics.mean(row['candidate_token_pairs_logical_only'] for row in group)})
    save(args.output / 'summary.json', {'started_utc': started, 'finished_utc': utc(),
        'runs_completed': len(results), 'summary': summary, 'results': results,
        'note': 'Synthetic indexer-only diagnostic. No real LM training, GPU benchmark, or proof of sparse-training efficiency.'})
    files = [p for p in args.output.rglob('*') if p.is_file()]
    save(args.output / 'manifest.json', [{'path': str(p.relative_to(args.output)).replace('\\', '/'), 'sha256': hash_file(p)} for p in files])


if __name__ == '__main__':
    main()
