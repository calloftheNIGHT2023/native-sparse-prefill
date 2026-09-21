"""Post-fit routing interventions. No optimizer steps; fixed-layout controls are not general methods."""
import json,math,shutil,sys,time,types
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from resume_joint_token_router import restore_checkpoint
from frozen_routing import select,routed_core
from run_frozen_router import now,save,sha,weights_sha

POLICIES=['original','other_row_scores','fixed_values','fixed_keys','prefix6','local8']

def positional_scores(n,policy,device):
    p=torch.arange(n,device=device);score=-p.float()
    if policy=='fixed_values':score[torch.tensor([1,3,5,7],device=device)]+=n*2
    elif policy=='fixed_keys':score[torch.tensor([0,2,4,6],device=device)]+=n*2
    elif policy=='local8':score=p.float()
    elif policy!='prefix6':raise ValueError(policy)
    return score.expand(n,n)

class Intervention:
    def __init__(self,m,ix,original_route,policy,active_layers=None):
        self.m=m;self.ix=ix;self.mode=original_route;self.policy=policy;self.originals=[];self.current=None
        self.active_layers=set(range(len(m.backbone.layers))) if active_layers is None else set(active_layers)
        self.stats=[dict(queries=0,source_hits=0,all_four_hits=0,edges_sum=0,max_edges=0) for _ in m.backbone.layers]
        for i,layer in enumerate(m.backbone.layers):
            self.originals.append(layer.sequence_mixer.forward);layer.sequence_mixer.forward=types.MethodType(self.forward(i),layer.sequence_mixer)
    def forward(self,i):
        def call(mha,x):
            q,k,v=mha.Wqkv(x).chunk(3,-1);n=x.shape[1];policy=self.policy if i in self.active_layers else 'original'
            if policy=='original':score=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1]) if self.mode=='exact' else self.ix[i](x)
            elif policy=='other_row_scores':score=self.ix[i](x).roll(1,0)
            else:score=positional_scores(n,policy,x.device).expand(len(x),n,n)
            mask=select(score);p=torch.arange(n,device=x.device)
            assert not mask.triu(1).any() and mask.sum(-1).max()<=8 and mask[:,p,p].all() and mask[:,p[1:],p[:-1]].all()
            b,pos,source=self.current;z=self.stats[i];z['queries']+=len(pos);z['source_hits']+=int(mask[b,pos,source].sum());z['all_four_hits']+=int(mask[b,pos][:,[1,3,5,7]].all(-1).sum());z['edges_sum']+=int(mask.sum());z['max_edges']=max(z['max_edges'],int(mask.sum(-1).max()))
            return mha.out_proj(routed_core(q,k,v,mask))
        return call
    def restore(self):
        for layer,fn in zip(self.m.backbone.layers,self.originals):layer.sequence_mixer.forward=fn

@torch.no_grad()
def evaluate(m,ix,mode,policy,data,active_layers=None):
    route=Intervention(m,ix,mode,policy,active_layers);predictions=[];correct=0;answers=0;nll=0.
    try:
        for first in range(0,len(data['inputs']),32):
            x=data['inputs'][first:first+32];y=data['labels'][first:first+32];b,p=torch.where(y!=-100)
            source=[next(j+1 for j in range(0,8,2) if x[bi,j]==x[bi,pi]) for bi,pi in zip(b.tolist(),p.tolist())]
            assert torch.equal(x[b,torch.tensor(source)],y[b,p]);route.current=(b,p,torch.tensor(source));logits=m(x);pred=logits.argmax(-1)[b,p];predictions+=pred.tolist();correct+=int((pred==y[b,p]).sum());answers+=len(p);nll+=float(torch.nn.functional.cross_entropy(logits[b,p],y[b,p],reduction='sum'))
    finally:route.restore()
    return dict(accuracy=correct/answers,answers=answers,nll=nll/answers,predictions=predictions,layers=[dict(**z,source_recall=z['source_hits']/z['queries'],all_four_coverage=z['all_four_hits']/z['queries']) for z in route.stats])

def main():
    torch.set_num_threads(4);src=ROOT/'results/joint-token-router-epoch80-v0';out=ROOT/'results/router-necessity-audit-v0';out.mkdir(exist_ok=False);start=now();timer=time.perf_counter();runs=[];count=0
    shutil.copy2(__file__,out/'source.py');shutil.copy2(ROOT/'docs/router-necessity-audit-plan-2026-09-14.md',out/'plan.md')
    data=torch.load(src/'new-test.pt',map_location='cpu',weights_only=True);prior=json.loads((src/'result.json').read_text())
    for row in prior['runs']:
        name=row['method'];parent=src/name/'checkpoint.pt';parent_sha=sha(parent);ck=torch.load(parent,map_location='cpu',weights_only=False);m,ix,route,*_=restore_checkpoint(ck,'cpu');route.restore();m.eval();before=weights_sha(m.state_dict());ibefore=weights_sha(ix.state_dict())
        for policy in POLICIES:
            r=dict(method=name,policy=policy,utc=now(),checkpoint_sha256=parent_sha)
            for key in ['fresh','swapped']:
                r[key]=evaluate(m,ix,ck['route'],policy,data[key]);count+=r[key]['answers']
                if policy=='original':assert r[key]['predictions']==row[key]['predictions']
            runs.append(r);save(out/(name+'__'+policy+'.json'),r);print(json.dumps(dict(method=name,policy=policy,fresh=r['fresh']['accuracy'],swapped=r['swapped']['accuracy'],layer1_all_four=r['fresh']['layers'][1]['all_four_coverage'])),flush=True)
        assert weights_sha(m.state_dict())==before and weights_sha(ix.state_dict())==ibefore and sha(parent)==parent_sha
    summary=dict(status='complete',started_utc=start,finished_utc=now(),wall_seconds=time.perf_counter()-timer,optimizer_updates=0,answers_evaluated=count,original_predictions_reproduced=15360,data_sha256=sha(src/'new-test.pt'),scope='Post-fit descriptive interventions on one seed, fixed layout; no causal exclusivity or novel method claim',runs=[dict(method=r['method'],policy=r['policy'],fresh=r['fresh']['accuracy'],swapped=r['swapped']['accuracy'],layers=r['fresh']['layers']) for r in runs])
    save(out/'summary.json',summary);save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(status='complete',optimizer_updates=0,answers=count,wall_seconds=summary['wall_seconds'])))

if __name__=='__main__':main()
