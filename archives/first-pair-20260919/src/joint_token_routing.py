"""Online KL indexer baseline; includes a dense detached teacher, no speed claim."""
import math,types
import torch
from torch.nn import functional as F
from frozen_routing import select,kl_loss

class JointTokenRouter:
    def __init__(self,model,indexers,route='learned'):
        assert route in ('learned','exact')
        self.model=model;self.indexers=indexers;self.route=route;self.originals=[];self.losses=[];self.collect_aux=True
        for i,layer in enumerate(model.backbone.layers):
            mha=layer.sequence_mixer;assert mha.num_heads==1
            self.originals.append(mha.forward);mha.forward=types.MethodType(self.forward_for(i),mha)
    def reset(self):self.losses=[]
    def forward_for(self,i):
        def forward(mha,x):
            q,k,v=mha.Wqkv(x).chunk(3,-1);n=x.shape[1];scores=self.indexers[i](x)
            # Full teacher is used during auxiliary fitting; no gradient to main Q/K or inputs.
            if self.collect_aux or self.route=='exact':
                with torch.no_grad():
                    teacher_scores=q.detach()@k.detach().transpose(-1,-2)/math.sqrt(q.shape[-1])
                    causal=torch.ones(n,n,device=x.device,dtype=torch.bool).tril()
                    target=teacher_scores.masked_fill(~causal,-torch.inf).softmax(-1)
                if self.collect_aux:self.losses.append(kl_loss(scores,target))
            mask=select(teacher_scores if self.route=='exact' else scores)
            ids=mask.to(torch.int32).topk(min(8,n),dim=-1).indices;valid=mask.gather(-1,ids)
            batch=torch.arange(len(q),device=x.device)[:,None,None];keys=k[batch,ids];values=v[batch,ids]
            selected=(q[:,:,None]*keys).sum(-1)/math.sqrt(q.shape[-1])
            attention=selected.masked_fill(~valid,-torch.inf).softmax(-1)
            attention=F.dropout(attention,p=mha.inner_attn.dropout_p if mha.training else 0.)
            return mha.out_proj((attention[:,:,:,None]*values).sum(-2))
        return forward
    def restore(self):
        for layer,f in zip(self.model.backbone.layers,self.originals):layer.sequence_mixer.forward=f
