"""Privileged head-selective routing diagnostic; not a deployable selector."""
import math
import torch
import run_flashmoba_realtext_precision as base
from task_route_intervention import metadata_from_ids, replace_question_blocks


def target_mass(q, k, scale, blocks, block=128):
    n, h, d = q.shape
    kv = k.shape[1]
    # Only final query; dense normalization is a diagnostic, not sparse speed.
    scores = torch.einsum('hgd,nhd->hgn', q[-1].float().reshape(kv, h // kv, d), k.float())
    probs = torch.softmax(scores.reshape(h, n) * scale, dim=-1)
    mask = torch.zeros(n, device=q.device, dtype=torch.bool)
    for b in blocks:
        mask[b * block:min((b + 1) * block, n)] = True
    mass = probs[:, mask].sum(-1)
    assert bool(torch.isfinite(mass).all())
    return mass.cpu().tolist()


def select_heads(mass, count=4):
    return sorted(range(len(mass)), key=lambda h: (-mass[h], h))[:count]


def selective_replace(ids, start, forced, heads, block=128):
    assert len(set(heads)) == len(heads) and all(0 <= h < ids.shape[1] for h in heads)
    changed = replace_question_blocks(ids, start, forced, block)
    result = ids.clone()
    result[:, heads] = changed[:, heads]
    other = [h for h in range(ids.shape[1]) if h not in heads]
    assert torch.equal(result[:, other], ids[:, other])
    assert torch.equal(result[:start], ids[:start])
    assert torch.equal((result >= 0).sum(-1), (ids >= 0).sum(-1))
    return result


def routed_attention(q, k, v, scale, mode, start, forced, target, heads=None, return_ids=False):
    n, h, d = q.shape
    block, topk = 128, 16
    nb = math.ceil(n / block)
    cu = torch.tensor([0, n], device=q.device, dtype=torch.int32)
    cm = torch.tensor([0, nb], device=q.device, dtype=torch.int32)
    inp = k.float()
    means = torch.zeros((nb, k.shape[1], d), device=k.device, dtype=torch.float32)
    base.mean_pool_kernel.fn[(nb, 1, k.shape[1])](
        inp, means, d, block, cu, cm, inp.stride(0), inp.stride(1), means.stride(0), means.stride(1),
        kBlockN=32, num_warps=4, num_stages=3)
    _, _, _, _, allids = base.flash_moba_cuda.moba_fused_topk(
        q, means.to(k.dtype), cu, cu, cm, n, n, topk, block, True)
    ids = allids[..., :topk].contiguous()
    mass = target_mass(q, k, scale, target) if mode == 'baseline' else None
    if mode != 'baseline':
        ids = selective_replace(ids, start, forced, heads, block)
    offsets, counts, indices = metadata_from_ids(ids, nb)
    out = base.flash_moba_attn_varlen_func(
        q, k, v, cu, cu, n, n, offsets, counts, indices,
        base.decide_lg_block_m(topk, block, n, True), block,
        dropout_p=0., softmax_scale=scale, causal=True)
    return (out, mass, ids) if return_ids else (out, mass)
