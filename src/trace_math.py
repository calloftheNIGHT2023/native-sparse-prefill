"""Dense CPU diagnostic math on frozen real-model attention traces."""
import math
import torch
from sparse_reference import expand_blocks


def prepare_trace(trace, layout):
    q, k, v = trace['q'], trace['k'], trace['v']
    logits = torch.einsum('thd,shd->hts', q, k) / math.sqrt(q.shape[-1])
    full_probs = logits.masked_fill(~layout.causal_mask[None], -torch.inf).softmax(-1)
    dense_out = full_probs @ v.permute(1, 0, 2)
    return {**trace, 'logits': logits, 'dense_out': dense_out,
            'full_target': target_from_logits(logits, layout.visible_blocks, layout)}


def target_from_logits(logits, support, layout):
    with torch.no_grad():
        mask = expand_blocks(support, layout)
        scores = logits.masked_fill(~mask[None], -torch.inf)
        scores[:, ~mask.any(-1)] = 0
        probabilities = scores.softmax(-1).mean(0) * mask
        pooled = probabilities[:, layout.block_tokens].amax(-1) * support
        return pooled / pooled.sum(-1, keepdim=True).clamp_min(1e-30)


def output_error(trace, selected, layout, late_start):
    mask = expand_blocks(selected, layout)
    probs = trace['logits'].masked_fill(~mask[None], -torch.inf).softmax(-1)
    output = probs @ trace['v'].permute(1, 0, 2)
    actual, expected = output[:, late_start:], trace['dense_out'][:, late_start:]
    return ((actual - expected).square().sum() / expected.square().sum().clamp_min(1e-30)).item()
