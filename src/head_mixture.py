"""Diagnostic sparse-vs-dense head mixture targets and sampled normalizers.

These identities are elementary conditional-probability algebra, not claimed
new theorems. All routines use dense cached logits for CPU correctness only.
"""
import torch
from sparse_reference import expand_blocks, sample_outside


def pool_target(probabilities, support, layout):
    pooled=probabilities.mean(0)[:,layout.block_tokens].amax(-1)*support
    return pooled/pooled.sum(-1,keepdim=True).clamp_min(1e-30)


def corrected_target(logits,support,layout,retained_mass):
    mask=expand_blocks(support,layout)
    local=logits.masked_fill(~mask[None],-torch.inf).softmax(-1)
    return pool_target(local*retained_mass[:,:,None],support,layout)


def dense_restricted_target(logits,support,layout):
    dense=logits.masked_fill(~layout.causal_mask[None],-torch.inf).softmax(-1)
    return pool_target(dense,support,layout)


def exact_retained_mass(logits,support,layout):
    full=logits.masked_fill(~layout.causal_mask[None],-torch.inf).softmax(-1)
    return (full*expand_blocks(support,layout)[None]).sum(-1)


def sampled_retained_mass(logits,support,layout,probe_blocks,generator,return_probes=False):
    probes=sample_outside(support,layout.visible_blocks,probe_blocks,generator)
    selected_mask=expand_blocks(support,layout)
    probe_mask=expand_blocks(probes,layout,include_tail=False)
    combined=selected_mask|probe_mask
    masked=logits.masked_fill(~combined[None],-torch.inf)
    shifted=(masked-masked.amax(-1,keepdim=True)).exp()
    selected_z=(shifted*selected_mask[None]).sum(-1)
    probe_z=(shifted*probe_mask[None]).sum(-1)
    outside=(layout.visible_blocks&~support).sum(-1)
    sampled=probes.sum(-1)
    factor=outside/sampled.clamp_min(1)
    z_est=selected_z+probe_z*factor[None]
    result=selected_z/z_est.clamp_min(1e-30)
    return (result,probes) if return_probes else result
