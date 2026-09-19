"""Exact-score top-k mechanism reference; deliberately not an efficient sparse kernel."""
import math
import torch
from torch import nn
from torch.nn import functional as F

SCHEDULES={'dense':(100,100),'native':(0,0),'all_warm4':(4,4),'first_warm8':(8,0),'second_warm8':(0,8)}

class ScheduledAttention(nn.Module):
    def __init__(self,layer_idx,schedule,k=8,local=2,dropout=0.1):
        super().__init__()
        if schedule not in SCHEDULES:raise ValueError(schedule)
        if k<local or local<1:raise ValueError('Require k >= local >= 1')
        self.layer_idx=layer_idx;self.schedule=schedule;self.k=k;self.local=local;self.dropout_p=dropout;self.epoch=0
        self.last_mask=None
    def dense(self):return self.epoch<SCHEDULES[self.schedule][self.layer_idx]
    def selection(self,scores):
        n=scores.shape[-1];p=torch.arange(n,device=scores.device)
        causal=p[:,None]>=p[None,:]
        if self.dense() or self.k>=n:return causal.expand_as(scores)
        local=causal & ((p[:,None]-p[None,:])<self.local)
        selected=local.expand_as(scores).clone()
        if self.k>self.local:
            ranking=scores.detach().masked_fill(~(causal&~local),float('-inf'))
            indices=ranking.topk(min(self.k-self.local,n),dim=-1).indices
            selected.scatter_(-1,indices,True)
        return selected & causal
    def forward(self,qkv):
        q,k,v=qkv.unbind(dim=2);n=q.shape[1]
        scores=torch.einsum('bthd,bshd->bhts',q,k/math.sqrt(q.shape[-1]))
        mask=self.selection(scores);self.last_mask=mask.detach()
        # Full shape retained for an identical dropout RNG layout in every condition.
        scores=scores.masked_fill(~mask,float('-inf'))
        attention=torch.softmax(scores,dim=-1,dtype=v.dtype)
        attention=F.dropout(attention,self.dropout_p if self.training else 0.)
        return torch.einsum('bhts,bshd->bthd',attention,v)

def install(model,schedule,k=8,local=2):
    for i,layer in enumerate(model.backbone.layers):
        old=layer.sequence_mixer.inner_attn
        layer.sequence_mixer.inner_attn=ScheduledAttention(i,schedule,k,local,old.dropout_p)

def set_epoch(model,epoch):
    for layer in model.backbone.layers:layer.sequence_mixer.inner_attn.epoch=epoch

def retained_edges(length,k):return sum(min(i+1,k) for i in range(length))
