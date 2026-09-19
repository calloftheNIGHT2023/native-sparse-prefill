"""Disposable eight-update measurements per execution path, outside scientific fits."""
import json,sys,time,statistics
from pathlib import Path
import torch
from zoology_entry import ROOT,set_determinism
from router_author_control import author_configs,selected_logits,generated
from router_author_sparse import build,initial_indexers,METHODS
from run_frozen_router import save,now,sha

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
out=ROOT/'results/router-author-sparse-preflight-v0';out.mkdir(exist_ok=False)
cfg=author_configs()[1];initial=torch.load(ROOT/'results/router-author-control-v0/initialization.pt',weights_only=True,map_location='cpu');ixstate=initial_indexers();d=generated(2026091621,256);x=d['inputs'].cuda();y=d['labels'].cuda();rows=[]
for method,epoch in [('warm4_selectedkl_r16',0),('warm4_selectedkl_r16',4),('ksa_additive_r16',0),('fixed_values6',0)]:
    m,ix,r=build(cfg,initial,ixstate,method,'cuda');r.epoch=epoch;m.train();opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.1);iopt=torch.optim.AdamW(ix.parameters(),lr=.001,weight_decay=0) if len(ix) else None;set_determinism(123);times=[];torch.cuda.reset_peak_memory_stats()
    for step in range(8):
        tick=time.perf_counter();opt.zero_grad()
        if iopt:iopt.zero_grad()
        r.reset();logits,target=selected_logits(m,x,y);loss=torch.nn.functional.cross_entropy(logits,target)+sum(r.losses);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in [*m.parameters(),*ix.parameters()] if p.grad is not None);opt.step()
        if iopt:iopt.step()
        torch.cuda.synchronize();times.append(time.perf_counter()-tick)
    rows.append(dict(method=method,epoch=epoch,technical_main_updates=8,technical_indexer_updates=8 if iopt else 0,median_seconds=statistics.median(times[3:]),all_seconds=times,peak_allocated_bytes=torch.cuda.max_memory_allocated(),cost=r.cost));r.restore();del opt,iopt,m,ix,r;torch.cuda.empty_cache()
est=(4*rows[0]['median_seconds']+15*rows[1]['median_seconds']+19*rows[2]['median_seconds']+19*rows[3]['median_seconds'])*391*1.25+180
result=dict(utc=now(),rows=rows,estimated_batch_seconds=est,scientific_launch_allowed=est<3600,technical_main_updates=32,technical_indexer_updates=24,scientific_updates=0,gpu=torch.cuda.get_device_name(),source_sha256=sha(Path(__file__)))
save(out/'result.json',result);print(json.dumps(result))
