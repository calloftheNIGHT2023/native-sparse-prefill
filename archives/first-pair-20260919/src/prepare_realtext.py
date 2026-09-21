"""Download pinned public assets and capture reproducible, verified attention traces."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import urllib.request

os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'

import pyarrow.parquet as pq
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def download(url, path):
    if path.exists():
        raise FileExistsError(f'Will not overwrite asset: {path}')
    partial = path.with_suffix(path.suffix + '.partial')
    if partial.exists():
        raise FileExistsError(f'Preserving partial transfer: {partial}')
    request = urllib.request.Request(url, headers={'User-Agent': 'NativeSparseResearch/0.1'})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as stream:
        shutil.copyfileobj(response, stream)
    partial.rename(path)
    return {'url': url, 'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'bytes': path.stat().st_size}


def main(args):
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Output exists; preserve prior data')
    out.mkdir(parents=True)
    shutil.copy2(args.config, out / 'frozen-config.json')
    shutil.copy2(Path(__file__), out / 'preparation-source.py')
    def event(kind, **extra):
        row = {'utc': utc(), 'event': kind, **extra}
        with (out / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    started = utc()
    event('start', device='cpu')
    try:
        assets, traces_dir = out / 'assets', out / 'traces'
        assets.mkdir()
        traces_dir.mkdir()
        model_dir = assets / 'model'
        model_dir.mkdir()
        downloaded = []
        cached_assets = {}
        if cfg.get('reuse_assets_from'):
            cached = json.loads((ROOT / cfg['reuse_assets_from'] / 'download-manifest.json').read_text(encoding='utf-8'))
            cached_assets = {item['url']: item for item in cached}
        def fetch_asset(url, path):
            if url not in cached_assets:
                return download(url, path)
            record = cached_assets[url]
            original = ROOT / record['path']
            if digest(original) != record['sha256']:
                raise ValueError('Cached source asset hash mismatch')
            if path.exists():
                raise FileExistsError('Preserve output asset')
            shutil.copy2(original, path)
            event('reuse_local_asset', url=url, original=str(original))
            return {**record, 'path': str(path.relative_to(ROOT)), 'reused_from': str(original)}
        for name in cfg.get('model_files', ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'README.md', 'model.safetensors']):
            url = f"https://huggingface.co/{cfg['model_id']}/resolve/{cfg['model_revision']}/{name}"
            event('download_start', file=name)
            downloaded.append(fetch_asset(url, model_dir / name))
            event('download_complete', file=name, bytes=downloaded[-1]['bytes'])
            save(out / 'download-manifest.json', downloaded)
        torch.set_num_threads(cfg['threads'])
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, attn_implementation='eager').float().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        seen_text, seen_prefix, examples = set(), set(), []
        for manifest_path in cfg.get('exclude_examples_from', []):
            previous = json.loads((ROOT / manifest_path).read_text(encoding='utf-8'))
            seen_text.update(r['text_sha256'] for r in previous['examples'])
            seen_prefix.update(r['token_prefix_sha256'] for r in previous['examples'])
            event('exclude_prior_examples', manifest=manifest_path, sha256=digest(ROOT / manifest_path))
        capture = {}
        hooks = []
        for layer_id in cfg['layer_ids']:
            def hook(module, inputs, kwargs, layer_id=layer_id):
                capture[layer_id] = {'hidden': inputs[0].detach().clone(),
                                     'rope': tuple(t.detach().clone() for t in kwargs['position_embeddings'])}
            hooks.append(model.gpt_neox.layers[layer_id].attention.register_forward_pre_hook(hook, with_kwargs=True))
        for split, count in cfg['examples'].items():
            datafile = assets / f'{split}.parquet'
            url = f"https://huggingface.co/datasets/{cfg['dataset_id']}/resolve/{cfg['dataset_revision']}/{cfg['dataset_config']}/{split}-00000-of-00001.parquet"
            downloaded.append(fetch_asset(url, datafile))
            save(out / 'download-manifest.json', downloaded)
            rows = pq.read_table(datafile, columns=['text']).column('text').to_pylist()
            order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(f"{cfg['selection_seed']}:{split}:{i}".encode()).digest())
            selected = 0
            for row_idx in order:
                text = rows[row_idx].strip()
                if not text or text.startswith('='):
                    continue
                full_hash = hashlib.sha256(text.encode()).hexdigest()
                if full_hash in seen_text:
                    continue
                ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
                if len(ids) < cfg['sequence_length'] + 1:
                    continue
                ids = ids[:cfg['sequence_length'] + 1]
                prefix_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                if prefix_hash in seen_prefix:
                    continue
                seen_text.add(full_hash)
                seen_prefix.add(prefix_hash)
                input_ids = torch.tensor([ids[:-1]])
                before = time.perf_counter()
                with torch.inference_mode():
                    output = model(input_ids=input_ids, use_cache=False, output_attentions=True)
                    layers = {}
                    for layer_id in cfg['layer_ids']:
                        layer = model.gpt_neox.layers[layer_id].attention
                        h, (cos, sin) = capture[layer_id]['hidden'], capture[layer_id]['rope']
                        qkv = layer.query_key_value(h).view(1, cfg['sequence_length'], model.config.num_attention_heads, 3 * layer.head_size).transpose(1, 2)
                        q, k, v = qkv.chunk(3, -1)
                        q, k = apply_rotary_pos_emb(q, k, cos, sin)
                        scores = (q @ k.transpose(-2, -1)) * layer.scaling
                        causal = torch.ones(cfg['sequence_length'], cfg['sequence_length'], dtype=torch.bool).tril()
                        attn = scores.masked_fill(~causal, -torch.inf).softmax(-1)
                        actual = output.attentions[layer_id]
                        discrepancy = (attn - actual).abs().max().item()
                        if discrepancy > 2e-5:
                            raise AssertionError(f'QK extraction mismatch: {discrepancy}')
                        # All stored data are detached CPU tensors, with no gradients or model state.
                        layers[str(layer_id)] = {'hidden': h[0].clone(), 'q': q[0].transpose(0, 1).clone(),
                            'k': k[0].transpose(0, 1).clone(), 'v': v[0].transpose(0, 1).clone(),
                            'qk_attention_max_abs_error': discrepancy}
                    dense_nll = torch.nn.functional.cross_entropy(output.logits[0], torch.tensor(ids[1:]), reduction='none')
                name = f'{split}-{selected:03d}'
                tensor_path = traces_dir / f'{name}.pt'
                torch.save({'input_ids': input_ids[0], 'next_ids': torch.tensor(ids[1:]),
                            'dense_token_nll': dense_nll.clone(), 'layers': layers}, tensor_path)
                example = {'id': name, 'split': split, 'row_index': row_idx,
                    'text_sha256': full_hash, 'token_prefix_sha256': prefix_hash,
                    'trace_path': f'traces/{name}.pt', 'trace_sha256': digest(tensor_path),
                    'source_tokens_before_truncation_at_least': cfg['sequence_length'] + 1,
                    'forward_seconds': time.perf_counter() - before,
                    'dense_nll': dense_nll.mean().item(),
                    'max_qk_attention_error': max(t['qk_attention_max_abs_error'] for t in layers.values())}
                examples.append(example)
                with (out / 'examples.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(example) + '\n')
                selected += 1
                event('trace_saved', id=name, max_qk_attention_error=example['max_qk_attention_error'])
                if selected == count:
                    break
            if selected != count:
                raise RuntimeError(f'Insufficient eligible paragraphs in {split}: {selected}/{count}')
        for hook in hooks:
            hook.remove()
        manifest = {'started_utc': started, 'finished_utc': utc(), 'config_sha256': digest(args.config),
            'examples': examples, 'counts': cfg['examples'], 'model_id': cfg['model_id'],
            'model_revision': cfg['model_revision'], 'data_revision': cfg['dataset_revision'],
            'parameters': sum(p.numel() for p in model.parameters()), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'device': 'cpu',
            'max_qk_attention_error': max(e['max_qk_attention_error'] for e in examples),
            'scope': 'Frozen dense pretrained-model traces on paragraphs; indexer split disjointness is not backbone pretraining decontamination.'}
        save(out / 'manifest.json', manifest)
        event('complete', examples=len(examples), parameters=manifest['parameters'])
    except Exception:
        event('failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/realtext-v0.json')
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
