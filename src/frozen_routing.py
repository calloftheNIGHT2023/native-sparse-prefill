"""Low-rank token-router capacity assay, not a reproduction of QSA/MSA kernels."""
import math,types
import torch
from torch import nn
from zoology_sparse_schedule import ScheduledAttention

def select(scores,k=8):
    return ScheduledAttention(0,'native',k=k,local=2,dropout=0).selection(scores)

class TokenIndexer(nn.Module):
    def __init__(self,width,rank):
        super().__init__();self.rank=rank;self.query=nn.Linear(width,rank);self.key=nn.Linear(width,rank)
    def forward(self,x):
        x=x.detach();return self.query(x)@self.key(x).transpose(-1,-2)/math.sqrt(self.rank)
    @torch.no_grad()
    def initialize(self,mha,mode):
        width=mha.d_model;w=mha.Wqkv.weight;b=mha.Wqkv.bias
        if mode=='copy':
            assert self.rank==width
            self.query.weight.copy_(w[:width]);self.key.weight.copy_(w[width:2*width])
            self.query.bias.copy_(b[:width]);self.key.bias.copy_(b[width:2*width]);return
        assert mode=='svd'
        q=torch.cat([w[:width].double().T,b[:width].double()[None]],0)
        k=torch.cat([w[width:2*width].double().T,b[width:2*width].double()[None]],0)
        u,s,vh=torch.linalg.svd(q@k.T/math.sqrt(width),full_matrices=False)
        q=u[:,:self.rank]*s[:self.rank].sqrt()*self.rank**.25
        k=vh[:self.rank].T*s[:self.rank].sqrt()*self.rank**.25
        self.query.weight.copy_(q[:-1].T);self.query.bias.copy_(q[-1]);self.key.weight.copy_(k[:-1].T);self.key.bias.copy_(k[-1])

def routed_core(q,k,v,mask):
    """Gather reference [B,N,D], selecting <=8 causal values; no main all-pairs QK."""
    count=min(8,q.shape[1]);ids=mask.to(torch.int32).topk(count,dim=-1).indices;valid=mask.gather(-1,ids)
    batch=torch.arange(len(q),device=q.device)[:,None,None]
    keys=k[batch,ids];values=v[batch,ids]
    logits=(q[:,:,None]*keys).sum(-1)/math.sqrt(q.shape[-1])
    probs=logits.masked_fill(~valid,-torch.inf).softmax(-1)
    return (probs[:,:,:,None]*values).sum(-2)

class FrozenRouter:
    def __init__(self,model,indexers):
        self.model=model;self.indexers=indexers;self.originals=[];self.coverage=[];self.exact_layers=set()
        for i,layer in enumerate(model.backbone.layers):
            mha=layer.sequence_mixer;assert mha.num_heads==1
            self.originals.append(mha.forward);mha.forward=types.MethodType(self.forward_for(i),mha)
    def forward_for(self,i):
        def forward(mha,x):
            if i in self.exact_layers:return self.originals[i](x)
            q,k,v=mha.Wqkv(x).chunk(3,-1);mask=select(self.indexers[i](x));output=routed_core(q,k,v,mask)
            return mha.out_proj(output)
        return forward
    def restore(self):
        for layer,f in zip(self.model.backbone.layers,self.originals):layer.sequence_mixer.forward=f

def full_teacher(mha,x):
    q,k,_=mha.Wqkv(x).chunk(3,-1);n=x.shape[1];mask=torch.ones(n,n,device=x.device,dtype=torch.bool).tril()
    return (q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])).masked_fill(~mask,-torch.inf).softmax(-1)

def kl_loss(scores,target):
    n=scores.shape[-1];causal=torch.ones(n,n,device=scores.device,dtype=torch.bool).tril()
    logp=scores.masked_fill(~causal,-torch.inf).log_softmax(-1).masked_fill(~causal,0.)
    safe_target=target.detach();return (safe_target*(safe_target.clamp_min(1e-30).log()-logp)).sum(-1).mean()
