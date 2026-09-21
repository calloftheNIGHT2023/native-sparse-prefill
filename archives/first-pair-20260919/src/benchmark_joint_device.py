"""Bounded device microbenchmark; never provisions a GPU or stops cloud billing.

Uses synthetic tokens, no scientific claims. CPU is the default. CUDA must be
explicitly requested. Main gather backend is a correctness reference only.
"""
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from transformers import AutoModelForCausalLM
from joint_attention import JointAttention
from run_joint_pilot import utc,save,sha

ROOT=Path(__file__).resolve().parents[1]

def main(args):
    if args.output.exists(): raise FileExistsError('Choose a fresh benchmark result')
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA runtime/device unavailable; CPU fallback is not a GPU test')
    torch.set_num_threads(4); torch.manual_seed(1440)
    cfg=json.loads((ROOT/'configs/joint-pilot-v0.json').read_text())
    cfg.update(sequence_length=args.length,late_query_start=args.length//2,
        selected_blocks=max(1,args.length//32),core_backend='gathered')
    model=AutoModelForCausalLM.from_pretrained(ROOT/cfg['pretrained_source'],local_files_only=True,
        trust_remote_code=False,use_safetensors=True,attn_implementation='eager').float().to(args.device)
    wrapper=JointAttention(model,cfg)
    params=list(model.parameters())+list(wrapper.indexers.parameters()); opt=torch.optim.AdamW(params,lr=1e-5)
    x=torch.randint(0,1000,(1,args.length),device=args.device)
    events=[]; started=utc(); start=time.perf_counter()
    if args.device=='cuda': torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(4):
            wrapper.reset('sparse',True); model.train(); opt.zero_grad(set_to_none=True)
            if args.device=='cuda': torch.cuda.synchronize()
            before=time.perf_counter()
            logits=model(x,use_cache=False).logits
            lm=torch.nn.functional.cross_entropy(logits[0,:-1],x[0,1:]); aux=torch.stack(wrapper.losses).mean()
            loss=lm+aux; loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True); opt.step()
            if args.device=='cuda': torch.cuda.synchronize()
            events.append({'utc':utc(),'step':step+1,'seconds':time.perf_counter()-before,
                'loss':float(loss),'gradient_norm':float(grad)})
            if time.perf_counter()-start>args.max_seconds: break
        result={'started_utc':started,'finished_utc':utc(),'device':args.device,'sequence_length':args.length,
            'events':events,'completed_updates':len(events),'tokens_are_synthetic':True,'scientific_runs':0,
            'max_seconds_limit':args.max_seconds,'limit_checked_between_steps':True,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
            'new_cloud_resources_created':0,'scope':'Reference gather full-model forward/backward; not a CUDA kernel speed comparison',
            'source_hashes':{name:sha(ROOT/'src'/name) for name in ['benchmark_joint_device.py','joint_attention.py','gathered_core.py','sparse_reference.py','routing_rules.py','indexer_calibration.py']}}
        save(args.output,result); print(json.dumps(result))
    finally: wrapper.restore()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--length',type=int,default=256); p.add_argument('--max-seconds',type=float,default=120)
    p.add_argument('--output',type=Path,required=True); main(p.parse_args())
