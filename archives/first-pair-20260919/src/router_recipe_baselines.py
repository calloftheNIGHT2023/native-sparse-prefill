"""Token-level adaptations of known warmup/selected-KL and KSA recipes, not new methods."""
import math,types
import torch
from torch.nn import functional as F
from frozen_routing import select,kl_loss

RECIPES=('warm4_selectedkl_r16','ksa_additive_r16')

class RecipeRouter:
    def __init__(self,model,indexers,recipe):
        assert recipe in RECIPES
        self.model=model;self.indexers=indexers;self.recipe=recipe;self.epoch=0;self.collect_aux=True;self.losses=[];self.originals=[];self.cost=[]
        for i,layer in enumerate(model.backbone.layers):
            mha=layer.sequence_mixer;assert mha.num_heads==1
            self.originals.append(mha.forward);mha.forward=types.MethodType(self.forward(i),mha)
    def reset(self):self.losses=[];self.cost=[]
    def forward(self,i):
        def call(mha,x):
            q,k,v=mha.Wqkv(x).chunk(3,-1);b,n,d=q.shape;ix=self.indexers[i];ksa=self.recipe=='ksa_additive_r16'
            scores=(ix.query(x)@ix.key(x).transpose(-1,-2)/math.sqrt(ix.rank)) if ksa else ix(x)
            warm=self.recipe=='warm4_selectedkl_r16' and self.epoch<4
            if warm:
                causal=torch.ones(n,n,device=x.device,dtype=torch.bool).tril();logits=q@k.transpose(-1,-2)/math.sqrt(d);attention=logits.masked_fill(~causal,-torch.inf).softmax(-1)
                if self.collect_aux:self.losses.append(kl_loss(scores,attention.detach()))
                output=F.dropout(attention,p=mha.inner_attn.dropout_p if mha.training else 0.)@v
                main_entries=b*n*n;teacher_entries=main_entries if self.collect_aux else 0
            else:
                mask=select(scores);ids=mask.to(torch.int32).topk(min(8,n),dim=-1).indices;valid=mask.gather(-1,ids);batch=torch.arange(b,device=x.device)[:,None,None]
                keys=k[batch,ids];values=v[batch,ids];main_logits=(q[:,:,None]*keys).sum(-1)/math.sqrt(d);s=scores.gather(-1,ids)
                if not ksa and self.collect_aux:
                    target=main_logits.detach().masked_fill(~valid,-torch.inf).softmax(-1)
                    logs=s.masked_fill(~valid,-torch.inf).log_softmax(-1).masked_fill(~valid,0.)
                    self.losses.append((target*(target.clamp_min(1e-30).log()-logs)).sum(-1).mean())
                attention=(main_logits+s if ksa else main_logits).masked_fill(~valid,-torch.inf).softmax(-1)
                attention=F.dropout(attention,p=mha.inner_attn.dropout_p if mha.training else 0.);output=(attention[:,:,:,None]*values).sum(-2)
                main_entries=b*n*min(8,n);teacher_entries=main_entries if (not ksa and self.collect_aux) else 0
            self.cost.append(dict(main_score_entries=main_entries,index_score_entries=b*n*n,teacher_distribution_entries=teacher_entries,dense_main=warm))
            return mha.out_proj(output)
        return call
    def restore(self):
        for layer,fn in zip(self.model.backbone.layers,self.originals):layer.sequence_mixer.forward=fn
