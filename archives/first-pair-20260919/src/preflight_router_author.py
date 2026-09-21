import json,math,statistics,time
from pathlib import Path
import torch
from zoology_entry import ROOT,set_determinism
from router_author_control import author_configs,make_model,generated,selected_logits,equivalence
from run_frozen_router import now,save

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
out=ROOT/'results/router-author-preflight-v0';out.mkdir(exist_ok=False)
correctness=equivalence('cuda');set_determinism(123);m=make_model(author_configs()[0],device='cuda');data=generated(2026091612,256);x=data['inputs'].cuda();y=data['labels'].cuda();opt=torch.optim.AdamW(m.parameters(),lr=author_configs()[0]['learning_rate'],weight_decay=.1);m.train();torch.cuda.reset_peak_memory_stats();records=[]
for step in range(8):
    torch.cuda.synchronize();tick=time.perf_counter();opt.zero_grad();logits,targets=selected_logits(m,x,y);loss=torch.nn.functional.cross_entropy(logits,targets);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);opt.step();torch.cuda.synchronize();records.append(dict(step=step+1,warmup=step<3,seconds=time.perf_counter()-tick,loss=float(loss.detach())))
median=statistics.median(r['seconds'] for r in records if not r['warmup']);estimate=median*391*64*1.25+120
result=dict(utc=now(),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda_equivalence=correctness,technical_main_updates=8,scientific_updates=0,parameters=sum(p.numel() for p in m.parameters()),batch=256,sequence_length=256,vocabulary=8192,peak_allocated_bytes=torch.cuda.max_memory_allocated(),median_step_seconds=median,estimated_max_64_epoch_seconds_with_margin=estimate,scientific_launch_allowed=estimate<2400,records=records)
save(out/'result.json',result);print(json.dumps(result))
