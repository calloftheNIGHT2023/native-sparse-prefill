"""Actual sparse forward/backward math check for K32/48/64, GQA14:2,128-token blocks."""
import torch
import torch.nn.functional as F
def check(mode,base):
 if mode==0:return dict(name='sparse_math',k=0,passed=True,scope='Dense flash SDPA previously validated; parent replay required')
 torch.manual_seed(2026091697)
 n=8704;h=14;hk=2;d=64;block=128;topk=mode
 q=torch.randn(n,h,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 k=torch.randn(n,hk,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 v=torch.randn(n,hk,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 out,ids,means=base.sparse(q,k,v,block,topk,'fp32',d**-.5,details=True)
 selected=ids[-16:].long();assert selected.shape==(16,h,topk) and bool((selected>=0).all()) and bool((selected<n//block).all())
 assert bool((selected==67).any(-1).all()) and bool((selected.sort(-1).values.diff(dim=-1)>0).all())
 # Last16 queries see68 blocks, so every requested K is genuinely sparse.
 qm=q.detach().float()[-16:];km=means.detach().float().repeat_interleave(h//hk,dim=1)
 scores=torch.einsum('qhd,bhd->qhb',qm,km);scores[:,:,67]=float('inf')
 reference_cut=scores.topk(topk,dim=-1).values[:,:,-1]
 selected_cut=scores.gather(-1,selected).min(-1).values
 regret=float((reference_cut-selected_cut).clamp_min(0).max());assert regret<.001
 probe=torch.randn_like(out[-16:],dtype=torch.float32)
 (out[-16:].float()*probe).sum().backward();got=[x.grad.detach().float().clone() for x in [q,k,v]]
 qf=q.detach().float().requires_grad_();kf=k.detach().float().requires_grad_();vf=v.detach().float().requires_grad_()
 keys=torch.arange(n,device='cuda');queries=torch.arange(n-16,n,device='cuda')
 mask=(keys[None,None,:,None]//block==selected.permute(1,0,2)[:,:,None,:]).any(-1)
 mask=mask & (keys[None,None,:]<=queries[None,:,None])
 logits=torch.einsum('qhd,khd->hqk',qf[-16:],kf.repeat_interleave(h//hk,dim=1))*(d**-.5)
 reference=torch.einsum('hqk,khd->qhd',logits.masked_fill(~mask,float('-inf')).softmax(-1),vf.repeat_interleave(h//hk,dim=1))
 output_rel=float((out[-16:].float()-reference.detach()).norm()/reference.detach().norm())
 (reference*probe).sum().backward();rels=[float((g-x.grad).norm()/x.grad.norm()) for g,x in zip(got,[qf,kf,vf])]
 passed=output_rel<.02 and max(rels)<.03
 assert passed,(output_rel,rels)
 return dict(name='genuine_sparse_math',k=topk,query_heads=h,kv_heads=hk,length=n,block=block,query_rows=16,selection_regret=regret,output_relative_l2=output_rel,gradient_relative_l2=rels,passed=passed,scope='Kernel selection vs FP32 gate scores; sparse output and q/k/v gradients vs explicit FP32 selected-block causal attention. Routing indices treated as discrete.')
