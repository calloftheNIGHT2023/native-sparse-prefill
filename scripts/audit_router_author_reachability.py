"""Exact two-layer fixed graph and paired invisible-value swaps; no optimizer updates."""
import json,shutil,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from router_pressure_models import fixed_mask
from router_author_sparse import build
from run_frozen_router import now,save,sha,weights_sha

@torch.no_grad()
def main():
    torch.set_num_threads(4);src=ROOT/'results/router-author-sparse-v0';out=ROOT/'results/router-author-reachability-v0';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py');shutil.copy2(ROOT/'docs/router-author-fixed-reachability-2026-09-14.md',out/'plan.md')
    mask=fixed_mask(256,'cpu')|torch.eye(256,dtype=torch.bool);reach=mask.to(torch.int32)@mask.to(torch.int32)>0;assert not reach.triu(1).any();torch.save(dict(mask=mask,reach=reach),out/'graph.pt')
    d=torch.load(src/'evaluation-data.pt',map_location='cpu',weights_only=True)['test'];x,y=d['inputs'],d['labels'];b,q=torch.where(y!=-100);bank=x[:,:32:2];source=(bank[b]==x[b,q,None]).to(torch.int32).argmax(-1)*2+1;visible=reach[q,source];mcount=reach[q][:,torch.arange(1,32,2)].sum(-1);upper=torch.where(visible,1.,1./(4096-mcount.double())).mean().item();result=json.loads((src/'fixed_values6/result.json').read_text(encoding='utf-8'));pred=torch.tensor(result['test']['predictions']);target=y[b,q]
    ck=torch.load(src/'fixed_values6/checkpoint.pt',map_location='cpu',weights_only=False);m,ix,route=build(ck['config'],ck['model'],ck['indexers'],'fixed_values6','cpu');m.eval();before=weights_sha(m.state_dict());new=x.clone();positions=[];chosen=[]
    for i in range(len(x)):
        candidates=torch.where((b==i)&~visible)[0]
        if not len(candidates):continue
        j=int(candidates[-1]);query=int(q[j]);s=int(source[j]);other=[v for v in range(1,32,2) if v!=s and not bool(reach[query,v])]
        if not other:continue
        t=other[0];new[i,s]=x[i,t];new[i,t]=x[i,s];assert not reach[query,s] and not reach[query,t] and x[i,s]!=x[i,t];chosen.append(i);positions.append(dict(row=i,query=query,source=s,other=t,old_target=int(x[i,s]),new_target=int(x[i,t])))
    maxerr=0.;same=0
    for first in range(0,len(chosen),32):
        ids=chosen[first:first+32];queries=torch.tensor([v['query'] for v in positions[first:first+32]]);a=m.lm_head(m.backbone(x[ids])[torch.arange(len(ids)),queries]);c=m.lm_head(m.backbone(new[ids])[torch.arange(len(ids)),queries]);error=float((a-c).abs().max());maxerr=max(maxerr,error);assert torch.equal(a,c);same+=len(ids)
    route.restore();assert weights_sha(m.state_dict())==before==result['final_backbone_sha'];save(out/'interventions.json',positions)
    summary=dict(utc=now(),status='complete',optimizer_updates=0,test_answers=len(target),reachable_answers=int(visible.sum()),unreachable_answers=int((~visible).sum()),reachable_fraction=float(visible.float().mean()),observed_source_values_min=int(mcount.min()),observed_source_values_max=int(mcount.max()),idealized_population_expectation_bound_evaluated_on_positions=upper,accuracy_reachable=float((pred[visible]==target[visible]).float().mean()),accuracy_unreachable=float((pred[~visible]==target[~visible]).float().mean()),invisible_swap_sequences=same,maximum_complete_logit_difference=maxerr,checkpoint_sha256=sha(src/'fixed_values6/checkpoint.pt'),scope='Fixed input-independent masks only. Conditional population-expectation upper bound, not a finite-sample ceiling, not a learned-router necessity claim.')
    save(out/'summary.json',summary);save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(summary))

if __name__=='__main__':main()
