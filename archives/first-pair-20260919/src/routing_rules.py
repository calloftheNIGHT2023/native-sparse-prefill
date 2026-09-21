"""Known local-context controls, not proposed new methods."""
import torch
from sparse_reference import select_blocks


def route_blocks(scores,layout,k,rule='learned'):
    visible=layout.visible_blocks
    if rule=='learned': return select_blocks(scores,visible,k)
    if rule not in ['learned_recent','sink_recent']: raise ValueError(rule)
    n,b=visible.shape
    if not b: return visible.clone()
    order=torch.arange(b,dtype=scores.dtype,device=scores.device).expand(n,-1)
    adjusted=scores.detach().clone() if rule=='learned_recent' else order.clone()
    if rule=='learned_recent':
        forced=order.masked_fill(~visible,-1).argmax(-1)
    else:
        forced=visible.to(torch.int32).argmax(-1)
    adjusted.scatter_(1,forced[:,None],torch.inf)
    return select_blocks(adjusted,visible,k)
