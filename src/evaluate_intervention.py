"""Frozen-model, one-layer-at-a-time NLL intervention for every saved indexer.

This measures prediction damage from a sparse mask in a small pretrained model.
No backbone training and no decode benchmark. Test paragraphs are the same as the
preceding diagnostic, so this is a secondary endpoint, not independent replication.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time
import traceback
import types

import torch
from transformers import AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from sparse_reference import QSAIndexer, make_layout, select_blocks, expand_blocks, teacher_block_distribution
from indexer_calibration import make_indexer


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def make_forward(indexer, rule, layout, cfg):
    def forward(module, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        if module.training or hidden_states.shape[0] != 1 or kwargs.get('layer_past') is not None:
            raise ValueError('Intervention is eval-only, batch 1, with no KV cache')
        _, n, hidden_dim = hidden_states.shape
        qkv = module.query_key_value(hidden_states).reshape(1, n, -1, 3 * module.head_size).transpose(1, 2)
        q, k, v = qkv.chunk(3, -1)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if rule == 'dense':
            selected = layout.visible_blocks
        elif rule == 'teacher_oracle':
            targets = teacher_block_distribution(q[0].transpose(0, 1), k[0].transpose(0, 1), layout.visible_blocks, layout)
            selected = select_blocks(targets, layout.visible_blocks, cfg['selected_blocks'])
        elif rule == 'sink_recent':
            scores = torch.arange(layout.visible_blocks.shape[-1]).float().expand(n, -1).clone()
            scores[:, 0] = scores.shape[-1] + 1
            selected = select_blocks(scores, layout.visible_blocks, cfg['selected_blocks'])
        else:
            selected = select_blocks(indexer(hidden_states[0], layout), layout.visible_blocks, cfg['selected_blocks'])
        mask = expand_blocks(selected, layout)
        scores = (q @ k.transpose(-2, -1)) * module.scaling
        scores = scores.masked_fill(~mask[None, None], torch.finfo(scores.dtype).min)
        probabilities = scores.softmax(-1)
        out = (probabilities @ v).transpose(1, 2).reshape(1, n, hidden_dim)
        return module.dense(out), probabilities
    return forward


def main(args):
    source, data, out = args.runs.resolve(), args.data.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError('Preserve prior intervention results')
    cfg = json.loads((source / 'frozen-config.json').read_text(encoding='utf-8'))
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True)
    shutil.copy2(Path(__file__), out / 'intervention-source.py')
    for source_name in ['sparse_reference.py', 'indexer_calibration.py']:
        shutil.copy2(Path(__file__).with_name(source_name), out / source_name)
    started = utc()
    def event(kind, **extra):
        with (out / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'utc': utc(), 'event': kind, **extra}) + '\n')
    event('start')
    try:
        torch.set_num_threads(cfg['threads'])
        model = AutoModelForCausalLM.from_pretrained(data / 'assets/model', local_files_only=True,
            trust_remote_code=False, use_safetensors=True, attn_implementation='eager').float().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        layout = make_layout([0] * cfg['sequence_length'], cfg['block_size'])
        tests = [(r['id'], torch.load(data / r['trace_path'], map_location='cpu', weights_only=True)) for r in manifest['examples'] if r['split'] == 'test']
        late = cfg['late_query_start']
        expected_dense = {name: t['dense_token_nll'][late:].mean().item() for name, t in tests}
        conditions = []
        for layer_id in cfg['layer_ids']:
            conditions.extend((layer_id, None, control) for control in ['dense', 'teacher_oracle', 'sink_recent'])
            conditions.extend((layer_id, r['seed'], r['method']) for r in summary['results'] if r['layer'] == layer_id)
        save(out / 'frozen-plan.json', {'created_utc': started, 'source_results': str(source), 'conditions': conditions,
            'metric': 'Mean next-token NLL in nats at positions 128..255, one changed attention layer',
            'test_set_reused': True, 'checkpoint_selection': 'fixed final checkpoint for all methods',
            'method_scope': 'No method selection by test NLL; all trained indexers and fixed controls are evaluated'})
        results = []
        for layer_id, seed, method in conditions:
            begin, timer = utc(), time.perf_counter()
            indexer = None
            if seed is not None:
                indexer = make_indexer(model.config.hidden_size, cfg).eval()
                weights = source / f'layer{layer_id}__seed{seed}__{method}/final-indexer.pt'
                indexer.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
            attention = model.gpt_neox.layers[layer_id].attention
            original = attention.forward
            attention.forward = types.MethodType(make_forward(indexer, method, layout, cfg), attention)
            per_example = []
            try:
                with torch.inference_mode():
                    for name, t in tests:
                        output = model(t['input_ids'][None], use_cache=False)
                        nll = torch.nn.functional.cross_entropy(output.logits[0], t['next_ids'], reduction='none')[late:].mean().item()
                        per_example.append({'id': name, 'nll': nll, 'dense_nll': expected_dense[name], 'delta_vs_dense': nll - expected_dense[name]})
            finally:
                attention.forward = original
            max_dense_difference = max(abs(r['delta_vs_dense']) for r in per_example) if method == 'dense' else None
            if max_dense_difference is not None and max_dense_difference > 1e-4:
                raise AssertionError(f'Dense intervention changed baseline NLL: {max_dense_difference}')
            row = {'layer': layer_id, 'seed': seed, 'method': method, 'started_utc': begin, 'finished_utc': utc(),
                'elapsed_seconds': time.perf_counter() - timer,
                'mean_nll': statistics.mean(r['nll'] for r in per_example),
                'mean_delta_vs_dense': statistics.mean(r['delta_vs_dense'] for r in per_example),
                'max_dense_difference': max_dense_difference, 'per_example': per_example}
            results.append(row)
            save(out / f'layer{layer_id}__seed{seed}__{method}.json', row)
            event('condition_complete', layer=layer_id, seed=seed, method=method, mean_nll=row['mean_nll'])
            print(json.dumps({k: row[k] for k in ['layer', 'seed', 'method', 'mean_nll', 'mean_delta_vs_dense']}), flush=True)
        save(out / 'summary.json', {'started_utc': started, 'finished_utc': utc(), 'results': results,
            'conditions': len(results), 'paragraphs_per_condition': len(tests),
            'scope': 'Frozen-model single-layer masking, secondary endpoint on reused test paragraphs. No training, long-context generalization, or GPU speed claim.'})
        event('complete', conditions=len(results))
        save(out / 'manifest.json', [{'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir() if p.is_file()])
    except Exception:
        event('failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
