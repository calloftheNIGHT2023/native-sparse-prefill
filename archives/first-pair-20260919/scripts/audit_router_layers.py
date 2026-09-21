import json,shutil,sys,time
from pathlib import Path
import torch
from audit_router_necessity import ROOT,evaluate
from resume_joint_token_router import restore_checkpoint
from run_frozen_router import now,save,sha,weights_sha

torch.set_num_threads(4);out=ROOT/'results/router-layer-necessity-audit-v0';out.mkdir(exist_ok=False);start=now();timer=time.perf_counter();src=ROOT/'results/joint-token-router-epoch80-v0';data=torch.load(src/'new-test.pt',map_location='cpu',weights_only=True);rows=[];count=0
for name in ['audit_router_necessity.py','audit_router_layers.py']:shutil.copy2(ROOT/'scripts'/name,out/name)
shutil.copy2(ROOT/'docs/router-layer-audit-plan-2026-09-14.md',out/'plan.md')
for name in ['exact_r16_shadow','learned_r16','learned_r64']:
    path=src/name/'checkpoint.pt';parent=sha(path);ck=torch.load(path,map_location='cpu',weights_only=False);m,ix,route,*_=restore_checkpoint(ck,'cpu');route.restore();m.eval();before=weights_sha(m.state_dict());ibefore=weights_sha(ix.state_dict())
    for layer in [0,1]:
        for policy in ['fixed_values','other_row_scores']:
            r=dict(method=name,layer=layer,policy=policy,utc=now(),checkpoint_sha256=parent)
            for key in ['fresh','swapped']:r[key]=evaluate(m,ix,ck['route'],policy,data[key],{layer});count+=r[key]['answers']
            save(out/f'{name}__layer{layer}__{policy}.json',r);row=dict(method=name,layer=layer,policy=policy,fresh=r['fresh']['accuracy'],swapped=r['swapped']['accuracy']);rows.append(row);print(json.dumps(row),flush=True)
    assert sha(path)==parent and weights_sha(m.state_dict())==before and weights_sha(ix.state_dict())==ibefore
summary=dict(status='complete',started_utc=start,finished_utc=now(),wall_seconds=time.perf_counter()-timer,optimizer_updates=0,answers_evaluated=count,data_sha256=sha(src/'new-test.pt'),rows=rows,scope='Post-hoc layer isolation after joint intervention, fixed weights and one seed; not a causal exclusivity proof')
save(out/'summary.json',summary);save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(status='complete',answers_evaluated=count,optimizer_updates=0)))
