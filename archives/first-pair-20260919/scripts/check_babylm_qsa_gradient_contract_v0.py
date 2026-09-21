"""CPU-only QSA gradient-contract reference, not a full Qwen/BabyLM model.
No training loop, optimizer update, CUDA work, model/tokenizer download, or timing claim.
"""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def layout(segments, block_size=4):
    n = len(segments)
    positions = torch.empty(n, dtype=torch.long)
    document = torch.empty(n, dtype=torch.long)
    blocks = []
    starts = [0] + [i for i in range(1, n) if segments[i] != segments[i - 1]]
    ends = starts[1:] + [n]
    for label, (start, end) in enumerate(zip(starts, ends)):
        positions[start:end] = torch.arange(end - start)
        document[start:end] = label
        blocks.extend([list(range(j, j + block_size)) for j in range(start, end - block_size + 1, block_size)])
    blocks = torch.tensor(blocks, dtype=torch.long).reshape(-1, block_size)
    causal = (document[:, None] == document[None, :]) & (torch.arange(n)[:, None] >= torch.arange(n)[None, :])
    visible = (document[:, None] == document[blocks[:, 0]][None, :]) & (torch.arange(n)[:, None] >= blocks[:, -1][None, :])
    tail = causal & (positions[None, :] >= (((positions + 1) // block_size) * block_size)[:, None])
    return {'blocks': blocks, 'visible': visible, 'tail': tail, 'causal': causal, 'positions': positions}


def rope(x, pos, dim=64, theta=1e7):
    # NeoX half pairing; block keys receive their block's starting position.
    phase = pos.to(x.dtype)[:, None] * theta ** (-torch.arange(0, dim, 2, dtype=x.dtype) / dim)[None, :]
    while phase.ndim < x.ndim:
        phase = phase.unsqueeze(1)
    a, b = x[..., :dim // 2], x[..., dim // 2:dim]
    return torch.cat((a * phase.cos() - b * phase.sin(), b * phase.cos() + a * phase.sin(), x[..., dim:]), dim=-1)


class Indexer(nn.Module):
    def __init__(self, hidden=32, heads=4, dim=128):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.q = nn.Linear(hidden, heads * dim, bias=False)
        self.k = nn.Linear(hidden, dim, bias=False)
        self.q_gain = nn.Parameter(torch.zeros(dim))
        self.k_gain = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden, meta):
        x = hidden.detach()
        q = self.q(x).reshape(len(x), self.heads, self.dim)
        projected_k = self.k(x)
        # Explicit FP32 pooling is a disclosed operation even in this float64 oracle.
        k = projected_k[meta['blocks']].float().mean(1).to(projected_k.dtype)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6) * (1 + self.q_gain)
        k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-6) * (1 + self.k_gain)
        q = rope(q, meta['positions'])
        k = rope(k, meta['positions'][meta['blocks'][:, 0]])
        return torch.relu(torch.einsum('thd,bd->thb', q, k)).sum(1)


def select(scores, meta, budget=2):
    chosen = torch.zeros_like(meta['visible'])
    if scores.shape[-1]:
        order = scores.detach().masked_fill(~meta['visible'], -torch.inf).argsort(dim=-1, descending=True, stable=True)
        chosen.scatter_(1, order[:, :min(budget, scores.shape[-1])], True)
    return chosen & meta['visible']


def support_tokens(chosen, meta):
    mask = meta['tail'].clone()
    mask[:, meta['blocks'].flatten()] |= chosen.repeat_interleave(4, dim=-1)
    return mask & meta['causal']


def core(q, k, v, chosen, meta):
    # q/k/v [tokens,heads,dim]. Dense scores are only the small CPU oracle.
    mask = support_tokens(chosen, meta)
    logits = torch.einsum('thd,shd->ths', q, k) / math.sqrt(q.shape[-1])
    probabilities = logits.masked_fill(~mask[:, None, :], -torch.inf).softmax(-1)
    output = torch.einsum('ths,shd->thd', probabilities, v)
    with torch.no_grad():
        pooled = probabilities.detach().mean(1)[:, meta['blocks']].amax(-1) * chosen
        teacher = pooled / pooled.sum(-1, keepdim=True).clamp_min(1e-30)
    return output, teacher


def selected_kl(scores, teacher, chosen):
    valid = chosen.any(-1)
    if not valid.any():
        return scores.sum() * 0
    mask = chosen[valid]
    log_p = scores[valid].masked_fill(~mask, -torch.inf).log_softmax(-1)
    log_p = torch.where(mask, log_p, torch.zeros_like(log_p))
    target = teacher[valid].detach() * mask
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-30)
    # All query positions count in N; empty rows contribute zero.
    return (target * (target.clamp_min(1e-30).log() - log_p)).sum() / scores.shape[0]


def main():
    torch.manual_seed(2026091701)
    torch.set_num_threads(2)
    n = 43
    meta = layout([0] * 21 + [1] * 19 + [0] * 3)
    indexer = Indexer().double()
    main_qkv = nn.Linear(32, 3 * 4 * 8, bias=False).double()
    lm_head = nn.Linear(32, 19, bias=False).double()
    hidden = torch.randn(n, 32, dtype=torch.float64, requires_grad=True)
    checks = {}

    scores = indexer(hidden, meta)
    scores.retain_grad()
    chosen = select(scores, meta)
    q, k, v = main_qkv(hidden).reshape(n, 3, 4, 8).unbind(1)
    q.retain_grad(); k.retain_grad(); v.retain_grad()
    output, target = core(q, k, v, chosen, meta)
    # A leaf target checks that selected_kl itself enforces target stop-gradient.
    target_leaf = target.detach().clone().requires_grad_(True)
    aux = selected_kl(scores, target_leaf, chosen)
    aux.backward()
    grad_norms = {name: float(p.grad.detach().norm()) if p.grad is not None else None for name, p in indexer.named_parameters()}
    assert all(value is not None and math.isfinite(value) and value > 0 for value in grad_norms.values())
    assert hidden.grad is None and main_qkv.weight.grad is None and q.grad is None and k.grad is None and v.grad is None
    assert target_leaf.grad is None and not target.requires_grad
    assert bool((scores.grad[~chosen] == 0).all())
    checks['selected_kl_gradient_isolation'] = {'pass': True, 'loss': float(aux.detach()), 'indexer_gradient_norms': grad_norms, 'teacher_requires_grad': target.requires_grad, 'target_leaf_grad_is_none': target_leaf.grad is None, 'hidden_and_main_grad_is_none': True, 'outside_score_gradient_zero': True}

    indexer.zero_grad(set_to_none=True)
    scores = indexer(hidden, meta)
    chosen = select(scores, meta)
    q, k, v = main_qkv(hidden).reshape(n, 3, 4, 8).unbind(1)
    output, _ = core(q, k, v, chosen, meta)
    labels = torch.arange(n) % 19
    lm_loss = F.cross_entropy(lm_head(output.reshape(n, 32)), labels)
    lm_loss.backward()
    assert all(p.grad is None for p in indexer.parameters())
    assert hidden.grad is not None and bool(hidden.grad.abs().sum() > 0)
    assert main_qkv.weight.grad is not None and bool(main_qkv.weight.grad.abs().sum() > 0)
    checks['lm_only_no_indexer_gradient'] = {'pass': True, 'lm_loss': float(lm_loss.detach()), 'hidden_grad_norm': float(hidden.grad.norm()), 'main_qkv_grad_norm': float(main_qkv.weight.grad.norm()), 'indexer_all_grad_none': True}

    @torch.no_grad()
    def evaluate(x):
        routes = select(indexer(x, meta), meta)
        qq, kk, vv = main_qkv(x).reshape(n, 3, 4, 8).unbind(1)
        out, _ = core(qq, kk, vv, routes, meta)
        return out, routes

    original, routes = evaluate(hidden.detach())
    mask = support_tokens(routes, meta)
    assert not bool((mask & ~meta['causal']).any())
    assert bool(mask.any(-1).all())
    for i in range(n):
        assert bool((meta['blocks'][routes[i]][:, -1] <= i).all())
    changed = hidden.detach().clone(); changed[10:21] += 100
    later, later_routes = evaluate(changed)
    torch.testing.assert_close(original[:10], later[:10], rtol=0, atol=0)
    assert torch.equal(routes[:10], later_routes[:10])
    changed = hidden.detach().clone(); changed[:21] -= 200
    other, other_routes = evaluate(changed)
    torch.testing.assert_close(original[21:], other[21:], rtol=0, atol=0)
    assert torch.equal(routes[21:], other_routes[21:])
    checks['microblock_causality_and_documents'] = {'pass': True, 'segments': [21, 19, 3], 'repeated_label_creates_new_segment': True, 'future_perturbation_exact_match': True, 'other_document_exact_match': True, 'kept_token_pairs': int(mask.sum())}

    boundary_meta = layout([0] * 260)
    boundary_scores = -torch.arange(65, dtype=torch.float64).expand(260, -1)
    boundary_routes = select(boundary_scores, boundary_meta, budget=64)
    boundary_mask = support_tokens(boundary_routes, boundary_meta)
    assert bool(boundary_mask[255, :256].all())
    assert bool(boundary_mask[256, :257].all())
    assert bool(boundary_mask[258, :259].all())
    assert int(boundary_mask[259].sum()) == 256
    assert not bool(boundary_routes[259, 64])
    assert not bool(boundary_mask[259, 256:260].any())
    checks['first_sparse_query_259_can_drop_current_complete_block'] = {'pass': True, 'query_index': 259, 'visible_blocks': 65, 'selected_blocks': 64, 'tail_tokens': 0, 'current_block_is_not_mandatory': True}

    # Empty and single-block support must have zero KL and score gradient.
    edge_scores = torch.tensor([[0.2, 0.3], [4., -2.]], dtype=torch.float64, requires_grad=True)
    edge_support = torch.tensor([[False, False], [True, False]])
    edge_loss = selected_kl(edge_scores, edge_support.double(), edge_support)
    edge_loss.backward()
    assert float(edge_loss.detach()) == 0 and bool((edge_scores.grad == 0).all())
    checks['empty_single_block_zero_kl'] = {'pass': True}

    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'cpu_gradient_contract_passed_not_full_model_validated', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'scientific_optimizer_updates': 0, 'training_tokens': 0, 'cpu_fixture_token_positions': n, 'device': 'cpu', 'torch': torch.__version__, 'checks': checks, 'limitations': ['Explicit dense score oracle: no efficiency claim.', 'Uses sparse-core detached teacher; does not reconstruct unpublished QSA training implementation.', 'This fixture is not a GDN/QSA language model or a BabyLM scientific run.', 'KL input hidden detach is a declared contract, not proven original training code fidelity.', 'Main fixture uses MHA for gradient isolation; new full model must separately validate GQA mapping, normalization, sigmoid gate, GDN boundaries, and optimizer state.', 'FP32 key pooling retained; no FP64 finite-difference claim through cast.']}
    destination = Path(__file__).resolve().parents[1] / 'results/babylm-qsa-gradient-contract-v0.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

