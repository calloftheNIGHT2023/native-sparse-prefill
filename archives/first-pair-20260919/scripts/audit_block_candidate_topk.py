"""Frozen full-model shortlist ablation; exposed MQAR, zero optimizer updates."""
import argparse
import gzip
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from router_author_control import make_model, selected_logits
from block_candidate_topk import BlockCandidateTopKAttention, block_summaries, coarse_scores, select_block_topk
from chunked_topk_attention import select_causal_topk


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x): p.write_text(json.dumps(x, indent=2), encoding='utf-8')


@torch.no_grad()
def routing_diagnostic(qkv, labels, configs):
    batch, n, _, heads, dim = qkv.shape
    q, k, v = [t.permute(0, 2, 1, 3).reshape(batch * heads, n, dim) for t in qkv.unbind(2)]
    ids = select_causal_topk(q, k, query_chunk=64)
    pos = torch.arange(n, device=q.device)
    scores = (q @ (k / math.sqrt(dim)).transpose(1, 2))
    remote = scores.masked_fill(pos[None, None, :] > pos[None, :, None] - 2, -torch.inf)
    cutoff = remote.topk(min(6, n), dim=-1).values[..., -1]
    valid = (ids >= 0)
    oracle_scores = scores.gather(-1, ids.clamp_min(0)).masked_fill(~valid, -torch.inf)
    weights = oracle_scores.softmax(-1)
    groups = torch.arange(len(q), device=q.device)[:, None, None]
    oracle_out = (weights[..., None] * v[groups, ids.clamp_min(0)]).sum(-2)
    mask_sets = {'all_late_queries': (pos >= n // 2)[None].expand(len(q), -1),
                 'answer_queries': (labels != -100)[:, None].expand(-1, heads, -1).reshape(len(q), n)}
    rows = []
    for cfg in configs:
        actual = select_block_topk(q, k, cfg['block'], cfg['routes'], cfg['method'], 32)
        recovered = ((ids[..., None] == actual[..., None, :]) & (actual[..., None, :] >= 0)).any(-1) & valid
        sw = scores.gather(-1, actual.clamp_min(0)).masked_fill(actual < 0, -torch.inf).softmax(-1)
        output = (sw[..., None] * v[groups, actual.clamp_min(0)]).sum(-2)
        block = cfg['block']; blocks = (n + block - 1) // block
        upper = coarse_scores(q, block_summaries(k, block), 'minmax')
        completed = torch.arange(blocks, device=q.device)[None, None, :] < (pos // block)[None, :, None]
        needs = (upper >= cutoff[..., None]) & completed
        # Optimistic: uses the true global sixth-remote cutoff for free; does not implement search.
        current_remote = ((pos - 2) - (pos // block) * block + 1).clamp_min(0)
        eligible_per_block = (pos[:, None] - 1 - torch.arange(blocks, device=q.device)[None, :] * block).clamp(0, block)
        certificate_fraction = ((needs * eligible_per_block).sum(-1) + current_remote) / (pos - 1).clamp_min(1)
        for name, mask in mask_sets.items():
            remote_valid = valid[..., 2:][mask]
            rows.append(dict(**cfg, query_set=name, queries=int(mask.sum()),
                remote_top6_recall=float(recovered[..., 2:][mask].sum() / remote_valid.sum()),
                retained_exact_top8_mass=float((recovered * weights).sum(-1)[mask].mean()),
                output_relative_squared_error=float((output - oracle_out)[mask].square().sum()
                    / oracle_out[mask].square().sum().clamp_min(1e-30)),
                oracle_cutoff_minmax_candidate_fraction=float(certificate_fraction[mask].mean())))
    return rows


def main(a):
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = utc(); tick = time.perf_counter()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from triton_topk_selector import TritonRankTopKAttention
    checkpoints = ['results/router-author-falsification-lowlr-v0/checkpoint.pt',
                   'results/router-author-falsification-confirm-v0/checkpoint.pt']
    folder = ROOT / 'results/router-author-falsification-evaluation-v0'
    data = torch.load(folder / 'evaluation-data.pt', weights_only=True, map_location='cpu')
    refs = json.loads((folder / 'evaluation.json').read_text())['conditions']
    sources = ['src/block_candidate_topk.py', 'scripts/audit_block_candidate_topk.py',
               'src/triton_selected_attention.py', 'src/triton_topk_selector.py',
               'src/chunked_topk_attention.py', 'src/router_author_control.py']
    protected = {p: sha(ROOT / p) for p in sources + checkpoints + [
        'results/router-author-falsification-evaluation-v0/evaluation-data.pt']}
    for p in sources:
        dest = out / 'source' / p; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / p).read_bytes())
    for source, path, checks in [
        ('src/triton_selected_attention.py', 'results/triton-selected-cuda-gate-v0/verification.json', 30),
        ('src/triton_topk_selector.py', 'results/triton-selector-cuda-gate-v0/verification.json', 15)]:
        gate = json.loads((ROOT / path).read_text())
        assert gate['status'] == 'passed' and len(gate['checks']) == checks
        assert gate['sources'][source] == protected[source]
    configs = [dict(method=m, block=b, routes=r) for m in ['mean', 'minmax']
               for b in [8, 16, 32] for r in [1, 2, 4]]
    save(out / 'manifest.json', dict(started_utc=started, sources=protected,
        configs=configs, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        scientific_optimizer_updates=0, selection_rule='2 local + 6 remote; completed blocks plus causal current block',
        plan=('Routing diagnostic recomputation only; fixes count of local keys in previous block.' if a.diagnostics_only else
              'All 18 configurations on both frozen models and all 3 exposed splits; no configuration removed for poor accuracy.'),
        data_scope='Previously exposed 1024-example MQAR splits at N256; diagnostic only'))
    rows = []; diagnostics = []
    def event(x):
        with (out / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(utc=utc(), elapsed_seconds=time.perf_counter() - tick, **x)) + '\n')
        print(json.dumps(x), flush=True)
    try:
        for path in checkpoints:
            ref = next(r for r in refs if r['checkpoint'] == path)
            assert sha(ROOT / path) == ref['checkpoint_sha256']
            ckpt = torch.load(ROOT / path, weights_only=False, map_location='cpu')
            model = make_model(ckpt['config'], ckpt['model'], 'cuda').float().eval()
            for layer in model.backbone.layers:
                layer.sequence_mixer.inner_attn = TritonRankTopKAttention(query_chunk=256).eval()
            for split in ['test', 'noisy']:
                captured = {}
                handles = []
                for layer_id, layer in enumerate(model.backbone.layers):
                    def hook(module, inputs, layer_id=layer_id):
                        captured[layer_id] = inputs[0].detach().clone()
                    handles.append(layer.sequence_mixer.inner_attn.register_forward_pre_hook(hook))
                with torch.no_grad():
                    selected_logits(model, data[split]['inputs'][:a.diagnostic_batch].cuda(),
                                    data[split]['labels'][:a.diagnostic_batch].cuda())
                for h in handles: h.remove()
                for layer_id, qkv in captured.items():
                    diag = routing_diagnostic(qkv, data[split]['labels'][:a.diagnostic_batch].cuda(), configs)
                    diagnostics.extend(dict(checkpoint=path, split=split, layer=layer_id, **r) for r in diag)
                save(out / 'routing-diagnostics.json', diagnostics)
                event(dict(event='diagnostics_complete', checkpoint=path, split=split))
            for cfg in ([] if a.diagnostics_only else [dict(method='exact', block=0, routes=0)] + configs):
                begin = utc(); condition_tick = time.perf_counter()
                for layer in model.backbone.layers:
                    layer.sequence_mixer.inner_attn = (TritonRankTopKAttention(query_chunk=256) if cfg['method'] == 'exact'
                        else BlockCandidateTopKAttention(**cfg, backend='triton', query_chunk=32)).eval()
                for split in ['test', 'swapped', 'noisy']:
                    predictions = []; correct_by_sequence = []; answers_by_sequence = []; losses = []
                    part = data[split]
                    with torch.no_grad():
                        for first in range(0, len(part['inputs']), a.batch):
                            logits, labels = selected_logits(model, part['inputs'][first:first+a.batch].cuda(),
                                part['labels'][first:first+a.batch].cuda())
                            assert bool(logits.isfinite().all())
                            pred = logits.argmax(-1); predictions.extend(pred.cpu().tolist())
                            ns = (part['labels'][first:first+a.batch] != -100).sum(-1).tolist()
                            count = 0
                            for number in ns:
                                correct_by_sequence.append(int((pred[count:count+number] == labels[count:count+number]).sum()))
                                answers_by_sequence.append(number); count += number
                            losses.append(float(torch.nn.functional.cross_entropy(logits, labels, reduction='sum')))
                    total = sum(answers_by_sequence); accuracy = sum(correct_by_sequence) / total
                    row = dict(checkpoint=path, **cfg, split=split, accuracy=accuracy,
                        answers=total, mean_nll=sum(losses)/total,
                        disagreements_from_archived=sum(x != y for x, y in zip(predictions, ref[split]['predictions'])),
                        correct_by_sequence=correct_by_sequence, answers_by_sequence=answers_by_sequence)
                    assert len(predictions) == len(ref[split]['predictions'])
                    if cfg['method'] == 'exact': assert row['disagreements_from_archived'] == 0
                    rows.append(row)
                    name = Path(path).parent.name + '-' + cfg['method'] + f"-b{cfg['block']}-r{cfg['routes']}-" + split
                    with gzip.open(out / (name + '-predictions.json.gz'), 'wt') as f: json.dump(predictions, f)
                    event({k: v for k, v in row.items() if not isinstance(v, list)})
                event(dict(event='condition_finished', checkpoint=path, **cfg,
                    started_utc=begin, finished_utc=utc(), wall_seconds=time.perf_counter()-condition_tick))
                save(out / 'partial-evaluation.json', rows)
                if time.perf_counter() - tick > a.max_seconds: raise TimeoutError('Stage wall cap')
            del model
        status = 'complete'; error = None
    except Exception as exc:
        status = 'incomplete'; error = dict(message=str(exc), traceback=traceback.format_exc())
        event(dict(event='failed', error=error))
    result = dict(status=status, error=error, started_utc=started, finished_utc=utc(),
        wall_seconds=time.perf_counter()-tick, conditions=rows, diagnostic_rows=len(diagnostics),
        scientific_optimizer_updates=0, sources=protected,
        protected_files_unchanged=all(sha(ROOT / p) == h for p, h in protected.items()),
        limitations=['Exploratory already-exposed toy data', 'Known coarse routing mechanisms, not a novelty claim',
            'QK pair reduction is not measured speedup', 'Certificate statistic uses oracle cutoff for free',
            'No larger language model training or end-to-end text quality evidence'])
    save(out / 'evaluation.json', result)
    if status != 'complete' or not result['protected_files_unchanged']: raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--batch', default=32, type=int); p.add_argument('--diagnostic-batch', default=16, type=int)
    p.add_argument('--max-seconds', default=900, type=float)
    p.add_argument('--diagnostics-only', action='store_true')
    main(p.parse_args())
