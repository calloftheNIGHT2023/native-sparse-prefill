"""Exploratory development-only attention descriptors; no causal claims or updates."""
import argparse,json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from zoology_entry import ROOT
from router_author_control import author_configs,evaluate
from run_router_author_falsification import model_for
from run_frozen_router import now,save,sha,weights_sha

@torch.no_grad()
def main(a):
    a.output.resolve().parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);p=a.checkpoint.resolve();c=torch.load(p,map_location='cpu',weights_only=False);m=model_for(author_configs()[1],c['model'],'cpu',a.method);m.eval();before=weights_sha(m.state_dict())
    data=torch.load(ROOT/'results/router-author-control-v0/data.pt',map_location='cpu',weights_only=True)['development'];d={k:v[:256] for k,v in data.items()};rows=[];current={};handles=[]
    for li,layer in enumerate(m.backbone.layers):
        def hook(module,args,output,li=li):
            q,k,v=args[0].unbind(dim=2);scores=torch.einsum('bthd,bshd->bhts',q,k/math.sqrt(q.shape[-1]));n=q.shape[1];pos=torch.arange(n);mask=module.selection(scores) if hasattr(module,'selection') else (pos[:,None]>=pos[None,:]).expand_as(scores)
            at=torch.softmax(scores.masked_fill(~mask,float('-inf')),dim=-1);reference=torch.einsum('bhts,bshd->bthd',at,v);torch.testing.assert_close(reference,output,rtol=2e-5,atol=2e-5)
            x=current['x'];y=current['y'];b,qp=torch.where(y!=-100);bank=x[b,1:32:2];target=(bank==y[b,qp,None]).int().argmax(-1)*2+1
            aa=at[:,0];mm=mask[:,0];values=torch.arange(1,32,2)
            rows.append(dict(layer=li+1,queries=len(b),value_to_previous_key_mass=float(aa[:,values,values-1].mean()),query_to_target_value_mass=float(aa[b,qp,target].mean()),query_to_any_source_value_mass=float(aa[b,qp][:,values].sum(-1).mean()),target_value_selected_fraction=float(mm[b,qp,target].float().mean()),query_attention_entropy=float((-(aa[b,qp]*(aa[b,qp]+1e-30).log()).sum(-1)).mean())))
        handles.append(layer.sequence_mixer.inner_attn.register_forward_hook(hook))
    for start in range(0,256,32):
        current.update(x=d['inputs'][start:start+32],y=d['labels'][start:start+32]);evaluate(m,{'inputs':current['x'],'labels':current['y']},'cpu',batch=32)
    for h in handles:h.remove()
    assert weights_sha(m.state_dict())==before
    layers=[]
    for li in [1,2]:
        rr=[r for r in rows if r['layer']==li];layers.append(dict(layer=li,**{k:sum(r[k] for r in rr)/len(rr) for k in rr[0] if k not in ['layer','queries']}))
    save(a.output.resolve(),dict(utc=now(),checkpoint=str(p),checkpoint_sha256=sha(p),method=a.method,epoch=c['epoch']+1,development_rows=256,layers=layers,optimizer_updates=0,scope='Exploratory descriptive attention on exposed development rows. Content-dependent selection is not a fixed causal graph; these are not impossibility proofs.'))
    print(json.dumps(layers),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--method',choices=['dense','exact_native'],required=True);p.add_argument('--output',required=True,type=Path);main(p.parse_args())
