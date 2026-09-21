"""Fixed-plan, CPU-only shared-indexer diagnostic on real Pythia Q/K/V traces."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback

import torch
from sparse_reference import QSAIndexer, make_layout, select_blocks, sample_outside, subset_kl, expand_blocks
from trace_math import prepare_trace, target_from_logits, output_error
from indexer_calibration import make_indexer
from head_mixture import dense_restricted_target, corrected_target, sampled_retained_mass

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(indexer, examples, layout, cfg, fixed=None):
    rows = []
    with torch.no_grad():
        for ex in examples:
            if fixed == 'dense':
                chosen = layout.visible_blocks
            elif fixed == 'teacher_oracle':
                chosen = select_blocks(ex['full_target'], layout.visible_blocks, cfg['selected_blocks'])
            elif fixed == 'sink_recent':
                scores = torch.arange(layout.visible_blocks.shape[1]).float()[None].expand(len(ex['hidden']), -1).clone()
                scores[:, 0] = scores.shape[1] + 1
                chosen = select_blocks(scores, layout.visible_blocks, cfg['selected_blocks'])
            else:
                scores = indexer(ex['hidden'], layout)
                chosen = select_blocks(scores, layout.visible_blocks, cfg['selected_blocks'])
            late = slice(cfg['late_query_start'], None)
            full = ex['full_target']
            top = full.argmax(-1)
            rows.append({'relative_output_error': output_error(ex, chosen, layout, cfg['late_query_start']),
                         'teacher_mass': (chosen * full).sum(-1)[late].mean().item(),
                         'teacher_top1_recall': chosen.gather(1, top[:, None]).squeeze(1)[late].float().mean().item()})
    return {k: statistics.mean(r[k] for r in rows) for k in rows[0]}


def run_one(cfg, layer_id, seed, method, examples, layout, folder):
    folder.mkdir()
    started, timer = utc(), time.perf_counter()
    def event(kind, **extra):
        with (folder / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'utc': utc(), 'monotonic_seconds': time.perf_counter() - timer, 'event': kind, **extra}) + '\n')
    event('start', layer=layer_id, seed=seed, method=method)
    try:
        torch.manual_seed(seed)
        idx = make_indexer(examples['train'][0]['hidden'].shape[-1], cfg)
        torch.save(idx.state_dict(), folder / 'initial-indexer.pt')
        probe_gen = torch.Generator().manual_seed(seed + 8888)
        # Deterministic cardinalities under fixed full-length, one-document inputs.
        dummy = select_blocks(torch.zeros_like(layout.visible_blocks, dtype=torch.float32), layout.visible_blocks, cfg['selected_blocks'])
        dummy_probe = sample_outside(dummy, layout.visible_blocks, cfg['probe_blocks'], torch.Generator().manual_seed(0))
        baseline_pairs = int(expand_blocks(dummy, layout).sum())
        probe_pairs = int(expand_blocks(dummy | dummy_probe, layout).sum())
        target_pair_budget = cfg['updates'] * probe_pairs
        updates = math.ceil(target_pair_budget / baseline_pairs) if method == 'selected_budget_match' else cfg['updates']
        optimizer = torch.optim.AdamW(idx.parameters(), lr=cfg['learning_rate'], weight_decay=0)
        curve = [{'step': 0, **evaluate(idx, examples['validation'], layout, cfg)}]
        event('development_eval', **curve[-1])
        elapsed, pairs = 0., 0
        for step in range(updates):
            t = time.perf_counter()
            ex = examples['train'][step % len(examples['train'])]
            scores = idx(ex['hidden'], layout)
            selected = select_blocks(scores, layout.visible_blocks, cfg['selected_blocks'])
            if method == 'full_supervision' or (method == 'dense_warmup' and step < cfg['warmup_updates']):
                support = layout.visible_blocks
            elif method == 'outside_probe':
                support = selected | sample_outside(selected, layout.visible_blocks, cfg['probe_blocks'], probe_gen)
            else:
                support = selected
            scored_support = support
            if method == 'normalizer_exact':
                target = dense_restricted_target(ex['logits'], support, layout)
                scored_support = layout.visible_blocks
            elif method == 'normalizer_sampled':
                mass, probes = sampled_retained_mass(ex['logits'], support, layout,
                    cfg['probe_blocks'], probe_gen, return_probes=True)
                target = corrected_target(ex['logits'], support, layout, mass)
                scored_support = support | probes
            else:
                target = target_from_logits(ex['logits'], support, layout)
            loss = subset_kl(scores, target, support)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite indexer objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(idx.parameters(), cfg['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            pairs += int(expand_blocks(scored_support, layout).sum())
            elapsed += time.perf_counter() - t
            event('optimizer_step', step=step + 1, loss=loss.item(), gradient_norm=norm.item(),
                  candidate_token_pairs=pairs, reference_train_seconds=elapsed)
            if (step + 1) % cfg['evaluation_every'] == 0 or step + 1 == updates:
                row = {'step': step + 1, 'candidate_token_pairs': pairs, 'reference_train_seconds': elapsed,
                       **evaluate(idx, examples['validation'], layout, cfg)}
                curve.append(row)
                event('development_eval', **row)
        final = evaluate(idx, examples['test'], layout, cfg)
        torch.save(idx.state_dict(), folder / 'final-indexer.pt')
        save(folder / 'development-curve.json', curve)
        result = {'layer': layer_id, 'seed': seed, 'method': method, 'updates': updates,
            'started_utc': started, 'finished_utc': utc(), 'elapsed_seconds': time.perf_counter() - timer,
            'reference_train_seconds': elapsed, 'candidate_token_pairs_logical_only': pairs,
            'probe_logical_pair_budget': target_pair_budget,
            'test': final, 'initial_development': curve[0], 'status': 'complete',
            'note': 'Frozen backbone; no LM updates. Logical query-token budget excludes index projections and is not GPU cost.'}
        save(folder / 'result.json', result)
        event('complete')
        print(json.dumps({'layer': layer_id, 'seed': seed, 'method': method, 'updates': updates, **final}), flush=True)
        return result
    except Exception:
        event('failed', traceback=traceback.format_exc())
        raise


def main(args):
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    out, data = args.output.resolve(), args.data.resolve()
    if out.exists():
        raise FileExistsError('Choose a new output folder, preserving previous runs')
    out.mkdir(parents=True)
    started = utc()
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    save(out / 'frozen-config.json', cfg)
    source_dir = out / 'source-snapshot'
    source_dir.mkdir()
    for name in ['run_realtext.py', 'trace_math.py', 'sparse_reference.py', 'indexer_calibration.py', 'head_mixture.py']:
        shutil.copy2(ROOT / 'src' / name, source_dir / name)
    source_manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    extraction_cfg = json.loads((data / 'frozen-config.json').read_text(encoding='utf-8'))
    for key in ['model_id', 'model_revision', 'dataset_id', 'dataset_revision', 'dataset_config',
                'sequence_length', 'layer_ids', 'examples', 'selection_seed', 'block_size']:
        if cfg[key] != extraction_cfg[key]:
            raise ValueError(f'Trace extraction config mismatch: {key}')
    records = source_manifest['examples']
    assert len({r['token_prefix_sha256'] for r in records}) == len(records)
    save(out / 'environment.json', {'started_utc': started, 'python': sys.version, 'torch': torch.__version__,
        'device': 'cpu', 'trace_manifest_sha256': sha(data / 'manifest.json'), 'data_path': str(data),
        'config_sha256': sha(args.config)})
    for record in records:
        assert sha(data / record['trace_path']) == record['trace_sha256']
    layout = make_layout([0] * cfg['sequence_length'], cfg['block_size'])
    results, controls = [], []
    for layer_id in cfg['layer_ids']:
        examples = defaultdict(list)
        for record in records:
            tensors = torch.load(data / record['trace_path'], map_location='cpu', weights_only=True)
            prepared = prepare_trace(tensors['layers'][str(layer_id)], layout)
            examples[record['split']].append(prepared)
        controls.append({'layer': layer_id, **{name: evaluate(None, examples['test'], layout, cfg, fixed=name)
                                             for name in ['dense', 'teacher_oracle', 'sink_recent']}})
        for seed in cfg['seeds']:
            for method in cfg['methods']:
                folder = out / f'layer{layer_id}__seed{seed}__{method}'
                results.append(run_one(cfg, layer_id, seed, method, examples, layout, folder))
    grouped = defaultdict(list)
    for row in results:
        grouped[(row['layer'], row['method'])].append(row)
    summary = []
    for (layer_id, method), group in grouped.items():
        keys = group[0]['test'].keys()
        summary.append({'layer': layer_id, 'method': method, 'seeds': len(group),
            'mean': {key: statistics.mean(r['test'][key] for r in group) for key in keys},
            'sample_std': {key: statistics.stdev(r['test'][key] for r in group) for key in keys},
            'logical_candidate_pairs': group[0]['candidate_token_pairs_logical_only']})
    gate_rows = []
    candidate = cfg.get('candidate_method', 'outside_probe')
    for layer_id in cfg['layer_ids']:
        by = {(r['seed'], r['method']): r['test']['relative_output_error'] for r in results if r['layer'] == layer_id}
        probe = statistics.mean(by[(s, candidate)] for s in cfg['seeds'])
        base = statistics.mean(by[(s, 'selected_only')] for s in cfg['seeds'])
        matched = statistics.mean(by[(s, 'selected_budget_match')] for s in cfg['seeds'])
        gate_rows.append({'layer': layer_id, 'relative_reduction_vs_selected': 1 - probe / base,
            'relative_reduction_vs_matched': 1 - probe / matched,
            'all_paired_seeds_better_than_selected': all(by[(s, candidate)] < by[(s, 'selected_only')] for s in cfg['seeds']),
            'candidate_method': candidate})
    passed = all(g['relative_reduction_vs_selected'] > .01 and g['relative_reduction_vs_matched'] > .01 and g['all_paired_seeds_better_than_selected'] for g in gate_rows)
    save(out / 'summary.json', {'started_utc': started, 'finished_utc': utc(), 'runs_completed': len(results),
        'summary': summary, 'controls': controls, 'gate': gate_rows, 'local_followup_gate_passed': passed,
        'results': results, 'note': cfg['scope']})
    save(out / 'manifest.json', [{'path': str(p.relative_to(out)).replace('\\', '/'), 'sha256': sha(p)} for p in out.rglob('*') if p.is_file()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/realtext-v0.json')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
