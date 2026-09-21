"""Pinned-author dense control and loss-equivalent supervised-position vocabulary head."""
import hashlib,json,math,runpy
from pathlib import Path
import numpy as np
import torch
from zoology_entry import ROOT,LanguageModel,multiquery_ar,set_determinism
from zoology.config import ModelConfig

SOURCE=ROOT/'literature/router-pressure-upstream-2026-09-14/zoology__experiments__paper_configs__iclr24_zoology_figure2__configs.py'
LRS=(0.002154434690031882,0.01)

def author_configs():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='074e53f43b2be0235c4e55547db3c9be3919323fc5edf1cf5a851c384b051262'
    cs=runpy.run_path(str(SOURCE))['configs'];out=[]
    for lr in LRS:
        choices=[c for c in cs if c.model.d_model==128 and c.model.sequence_mixer.name=='zoology.mixers.attention.MHA' and c.data.train_configs[0].input_seq_len==256 and c.learning_rate==lr];assert len(choices)==1
        c=choices[0];assert c.data.train_configs[0].num_kv_pairs==16 and not c.data.train_configs[0].random_non_queries and c.model.state_mixer.name=='torch.nn.Identity'
        out.append(c.model_dump(serialize_as_any=True))
    return out

def make_model(config,initial=None,device='cpu'):
    m=LanguageModel(ModelConfig(**config['model']))
    if initial is not None:m.load_state_dict(initial)
    m.to(device);m.backbone.embeddings.device=device;return m

def selected_logits(model,x,y):
    mask=y!=-100;hidden=model.backbone(x);return model.lm_head(hidden[mask]),y[mask]

def generated(seed,rows):
    nr=np.random.get_state()
    try:
        s=multiquery_ar(vocab_size=8192,num_examples=rows,input_seq_len=256,num_kv_pairs=16,random_non_queries=False,seed=seed)
        return dict(inputs=s.inputs,labels=s.labels)
    finally:np.random.set_state(nr)

def validate(data,clean=True):
    x,y=data['inputs'],data['labels'];assert x.shape==y.shape and x.shape[1]==256;assert torch.all((y!=-100).sum(-1)==16)
    bank=x[:,:32:2];values=x[:,1:32:2];assert torch.all(bank.sort(-1).values[:,1:]!=bank.sort(-1).values[:,:-1]);assert torch.all(values.sort(-1).values[:,1:]!=values.sort(-1).values[:,:-1])
    b,q=torch.where(y!=-100);matches=bank[b]==x[b,q,None];assert torch.all(q>=32) and torch.all(matches.sum(-1)==1);assert torch.equal(values[b,matches.to(torch.int32).argmax(-1)],y[b,q])
    if clean:assert torch.all(x[:,32:][y[:,32:]==-100]==0)

def row_hashes(data):return {hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in data['inputs']}

def interventions(test):
    x,y=test['inputs'],test['labels'];swap=dict(inputs=x.clone(),labels=torch.full_like(y,-100));pairs=[]
    for i in range(len(x)):
        q=int(torch.where(y[i]!=-100)[0][-1]);s=int(torch.where(x[i,:32:2]==x[i,q])[0][0])*2+1;t=(s+2)%32
        swap['inputs'][i,s]=x[i,t];swap['inputs'][i,t]=x[i,s];swap['labels'][i,q]=x[i,t];pairs.append([q,s,t])
    noisy=dict(inputs=x.clone(),labels=y.clone());mask=noisy['inputs']==0;fill=torch.randint(0,8192,x.shape,generator=torch.Generator().manual_seed(2026091602));noisy['inputs'][mask]=fill[mask];validate(noisy,False)
    assert torch.equal(noisy['inputs'][:,:32],x[:,:32]) and torch.equal(noisy['inputs'][y!=-100],x[y!=-100]);return swap,noisy,pairs

@torch.no_grad()
def evaluate(model,data,device,batch=256):
    model.eval();predictions=[];nll=0.;correct=0;count=0
    for first in range(0,len(data['inputs']),batch):
        x=data['inputs'][first:first+batch].to(device);y=data['labels'][first:first+batch].to(device);logits,target=selected_logits(model,x,y);p=logits.argmax(-1);predictions.extend(p.cpu().tolist());correct+=int((p==target).sum());count+=len(target);nll+=float(torch.nn.functional.cross_entropy(logits,target,reduction='sum'))
    return dict(accuracy=correct/count,answers=count,nll=nll/count,predictions=predictions)

def equivalence(device):
    cfg=author_configs()[0];set_determinism(2026091600);a=make_model(cfg,device=device);b=make_model(cfg,a.state_dict(),device);data=generated(2026091610,2);x=data['inputs'].to(device);y=data['labels'].to(device)
    a.train();b.train();set_determinism(7781);full=a(x);la=torch.nn.functional.cross_entropy(full.flatten(0,1),y.flatten());la.backward()
    set_determinism(7781);small,target=selected_logits(b,x,y);lb=torch.nn.functional.cross_entropy(small,target);lb.backward();torch.testing.assert_close(small,full[y!=-100],rtol=5e-5,atol=3e-6);torch.testing.assert_close(la,lb,rtol=1e-5,atol=1e-6)
    error=0.
    for p,q in zip(a.parameters(),b.parameters()):
        assert p.grad is not None and q.grad is not None;torch.testing.assert_close(p.grad,q.grad,rtol=4e-4,atol=3e-6);error=max(error,float((p.grad-q.grad).abs().max()))
    assert a.lm_head.weight is a.backbone.embeddings.word_embeddings.weight and b.lm_head.weight is b.backbone.embeddings.word_embeddings.weight
    return dict(device=device,loss_full=float(la.detach()),loss_selected=float(lb.detach()),maximum_gradient_error=error,optimizer_updates=0)
