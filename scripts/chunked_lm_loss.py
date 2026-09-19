"""Backpropagate a frozen output head in token chunks, then traverse the backbone once."""
from contextlib import nullcontext
import torch
import torch.nn.functional as F

def chunked_head_backward(hidden, head, targets, chunk_size, amp_context=nullcontext):
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError('This pilot supports batch size one only.')
    if targets.shape != (hidden.shape[1],) or chunk_size <= 0:
        raise ValueError('Targets must have one entry per position; positive chunk size required.')
    if any(p.requires_grad for p in head.parameters()):
        raise ValueError('The output head must be frozen for this pilot.')
    leaf = hidden.detach().requires_grad_(True)
    n = hidden.shape[1]
    total = torch.zeros((), device=hidden.device, dtype=torch.float32 if hidden.dtype!=torch.float64 else torch.float64)
    for start in range(0,n,chunk_size):
        end=min(start+chunk_size,n)
        with amp_context():
            logits=head(leaf[:,start:end])
            # Preserve FP64 for the independent CPU check; use stable FP32 CE in AMP.
            ce_logits=logits if logits.dtype==torch.float64 else logits.float()
            loss=F.cross_entropy(ce_logits[0],targets[start:end],reduction='sum')/n
        loss.backward()
        total.add_(loss.detach())
        del logits,ce_logits,loss
    torch.autograd.backward(hidden,leaf.grad)
    return total.detach()
