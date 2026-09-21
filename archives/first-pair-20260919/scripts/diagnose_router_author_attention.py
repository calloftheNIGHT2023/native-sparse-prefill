"""Descriptive attention masses at initialization and final dense author checkpoints."""
import json,math,shutil,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from router_author_control import make_model
from run_frozen_router import now,save,sha,weights_sha

@torch.no_grad()
def measure(m,d):
    m.eval();sums=[dict(value_previous_key=0.,query_source_value=0.,query_source_key=0.,query_all_source_values=0.) for _ in range(2)];context={};hooks=[]
    def make_hook(i):
        def hook(module,args):
            h=args[0];q,k,_=module.Wqkv(h).chunk(3,-1);n=h.shape[1];p=(q@(k/math.sqrt(h.shape[-1])).transpose(-1,-2)).masked_fill(~torch.ones(n,n,dtype=torch.bool).tril(),-torch.inf).softmax(-1)
            x,y=context['x'],context['y'];b,pos=torch.where(y!=-100);source=(x[b,:32:2]==x[b,pos,None]).to(torch.int32).argmax(-1)*2+1;values=torch.arange(1,32,2)
            sums[i]['value_previous_key']+=float(p[:,values,values-1].double().sum());sums[i]['query_source_value']+=float(p[b,pos,source].double().sum());sums[i]['query_source_key']+=float(p[b,pos,source-1].double().sum());sums[i]['query_all_source_values']+=float(p[b,pos][:,values].double().sum())
        return hook
    for i,layer in enumerate(m.backbone.layers):hooks.append(layer.sequence_mixer.register_forward_pre_hook(make_hook(i)))
    for first in range(0,len(d['inputs']),32):
        context.update(x=d['inputs'][first:first+32],y=d['labels'][first:first+32]);m.backbone(context['x'])
    for h in hooks:h.remove()
    return [{k:v/(16*len(d['inputs'])) for k,v in s.items()} for s in sums]

def main():
    torch.set_num_threads(4);src=ROOT/'results/router-author-control-v0';out=ROOT/'results/router-author-attention-v0';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py');shutil.copy2(ROOT/'docs/router-author-attention-diagnostic-2026-09-14.md',out/'plan.md')
    r=json.loads((src/'result.json').read_text());cfgs=json.loads((src/'resolved-configs.json').read_text());d=torch.load(src/'data.pt',map_location='cpu',weights_only=True)['test'];initial=torch.load(src/'initialization.pt',map_location='cpu',weights_only=True);conditions=[('initialization',cfgs[0],initial)]
    for run in r['runs']:
        c=torch.load(src/run['method']/'checkpoint.pt',map_location='cpu',weights_only=False);conditions.append((run['method'],c['config'],c['model']))
    rows=[]
    for name,cfg,state in conditions:
        m=make_model(cfg,state);before=weights_sha(m.state_dict());stats=measure(m,d);assert before==weights_sha(m.state_dict());rows.append(dict(condition=name,layers=stats,query_answers=16384,source_value_positions=16384))
    result=dict(utc=now(),status='complete',optimizer_updates=0,conditions=rows,scope='Descriptive attention mass, not a causal-necessity test or a new mechanism claim.');save(out/'summary.json',result);save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(result))

if __name__=='__main__':main()
