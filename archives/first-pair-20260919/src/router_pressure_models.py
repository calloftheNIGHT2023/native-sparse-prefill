"""Shared pressure-test model wrappers. Fixed masks know the public synthetic layout."""
import math,types
import torch
from torch.nn import functional as F
from run_joint_token_router import make_model
from frozen_routing import TokenIndexer,select
from router_recipe_baselines import RecipeRouter,RECIPES

METHODS=('dense',*RECIPES,'fixed_values6')

def fixed_mask(n,device):
    p=torch.arange(n,device=device);priority=-p.float();fav=p[(p<12)&(p%2==1)];priority[fav]+=2*n
    return select(priority.expand(n,n))

class ControlRouter:
    def __init__(self,model,method):
        assert method in ('dense','fixed_values6')
        self.model=model;self.method=method;self.epoch=0;self.collect_aux=False;self.losses=[];self.cost=[];self.originals=[]
        for layer in model.backbone.layers:
            mha=layer.sequence_mixer;self.originals.append(mha.forward);mha.forward=types.MethodType(self.forward,mha)
    def reset(self):self.losses=[];self.cost=[]
    def forward(self,mha,x):
        q,k,v=mha.Wqkv(x).chunk(3,-1);b,n,d=q.shape
        if self.method=='dense':
            mask=torch.ones(n,n,device=x.device,dtype=torch.bool).tril();scores=q@k.transpose(-1,-2)/math.sqrt(d);prob=scores.masked_fill(~mask,-torch.inf).softmax(-1);prob=F.dropout(prob,p=mha.inner_attn.dropout_p if mha.training else 0.);out=prob@v;entries=b*n*n
        else:
            mask=fixed_mask(n,x.device).expand(b,n,n);ids=mask.to(torch.int32).topk(min(8,n),dim=-1).indices;valid=mask.gather(-1,ids);batch=torch.arange(b,device=x.device)[:,None,None]
            keys=k[batch,ids];values=v[batch,ids];scores=(q[:,:,None]*keys).sum(-1)/math.sqrt(d);prob=scores.masked_fill(~valid,-torch.inf).softmax(-1);prob=F.dropout(prob,p=mha.inner_attn.dropout_p if mha.training else 0.);out=(prob[:,:,:,None]*values).sum(-2);entries=b*n*min(8,n)
        self.cost.append(dict(main_score_entries=entries,index_score_entries=0,teacher_distribution_entries=0,dense_main=self.method=='dense'))
        return mha.out_proj(out)
    def restore(self):
        for layer,fn in zip(self.model.backbone.layers,self.originals):layer.sequence_mixer.forward=fn

def build(base,main_state,index_state,method,device):
    m=make_model(base,main_state,device);ix=torch.nn.ModuleList([TokenIndexer(128,16) for _ in range(2)] if method in RECIPES else []).to(device)
    if method in RECIPES:ix.load_state_dict(index_state);route=RecipeRouter(m,ix,method)
    else:route=ControlRouter(m,method)
    return m,ix,route

def load_checkpoint(ck,device):
    m,ix,r=build(ck['config'],ck['model'],ck['indexers'],ck['method'],device);r.epoch=ck['epoch']+1;r.collect_aux=False;m.eval();return m,ix,r
