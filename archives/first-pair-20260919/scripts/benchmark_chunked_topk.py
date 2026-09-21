"""Bounded attention-core benchmarks. No training, rental or billing operations.

Primary sparse timings INCLUDE exact selection. Cached-index component timings
are explicitly diagnostic. CPU measurements are not GPU speed evidence. Dense
SDPA runs without a quadratic user mask; forced CUDA Flash never silently falls
back. All forward/backward trials use the same tensor shapes and dtype.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from chunked_topk_attention import ChunkedTopKAttention,select_causal_topk,selected_attention,operation_accounting
from zoology_sparse_schedule import ScheduledAttention


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


class DenseSDPA(nn.Module):
    def __init__(self,dropout,force_flash=False):
        super().__init__();self.dropout=dropout;self.force_flash=force_flash
    def forward(self,qkv):
        q,k,v=[a.permute(0,2,1,3) for a in qkv.unbind(2)]
        if self.force_flash:
            from torch.nn.attention import sdpa_kernel,SDPBackend
            ctx=sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
        else:ctx=contextlib.nullcontext()
        with ctx:
            return F.scaled_dot_product_attention(q,k,v,is_causal=True,
                dropout_p=self.dropout if self.training else 0.0).permute(0,2,1,3)


def saved_storage(model,x,go):
    """Autograd saved storage, not allocator peak; counting deduplicates views."""
    seen={};shapes=[];logical=0
    def pack(t):
        nonlocal logical
        logical+=t.numel()*t.element_size();shapes.append(list(t.shape))
        storage=t.untyped_storage()
        seen[(str(t.device),storage.data_ptr())]=storage.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        y=model(x)
    torch.autograd.grad(y,x,go)
    return dict(logical_saved_tensor_bytes=logical,unique_saved_storage_bytes=sum(seen.values()),
                saved_tensor_shapes=shapes,
                definition='Includes input storage referenced by autograd; excludes transient workspace and allocator cache.')


def main(a):
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    if min(a.lengths+[a.batch,a.heads,a.dim,a.query_chunk,a.repeats,a.warmup])<1:
        raise ValueError('Lengths, shapes, chunks and repeat counts must be positive')
    if not 1<=a.local<=a.budget or not 0<=a.dropout<1:raise ValueError('Invalid sparsity/dropout')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dtype=getattr(torch,a.dtype);is_cuda=a.device=='cuda'
    device_details=dict(device=a.device,torch_version=torch.__version__,python=sys.version,
        platform=platform.platform(),threads=a.threads,cuda_runtime=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name() if is_cuda else None,
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory if is_cuda else None)
    if is_cuda:
        device_details['free_total_bytes_before_inputs']=list(torch.cuda.mem_get_info())
        try:
            observed=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu',
                '--format=csv,noheader'],capture_output=True,text=True,timeout=5)
            device_details['gpu_snapshot_before_inputs']=observed.stdout.strip()
        except Exception as e:device_details['gpu_snapshot_error']=str(e)
        device_details['exclusive_gpu_access_verified']=False
    paths=[Path(__file__),ROOT/'src/chunked_topk_attention.py',ROOT/'src/zoology_sparse_schedule.py']
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(out/'manifest.json',dict(utc=now(),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        environment=device_details,sources=sources,scientific_optimizer_updates=0,
        scope='One attention core after Q/K/V projection, not full-model training or prompt prefill latency'))
    for p in paths:
        dst=out/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    started=now();tick=time.perf_counter();rows=[];components=[]
    def sync():
        if is_cuda:torch.cuda.synchronize()
    def deadline():
        if time.perf_counter()-tick>a.max_seconds:raise TimeoutError('Benchmark wall cap, checked between operations')
    def record(event):
        event=dict(utc=now(),**event)
        with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
    def samples(fn):
        for _ in range(a.warmup):deadline();fn();sync()
        gc.collect();sync()
        baseline=torch.cuda.memory_allocated() if is_cuda else None
        if is_cuda:torch.cuda.reset_peak_memory_stats()
        values=[]
        for repeat in range(a.repeats):
            deadline();sync();t=time.perf_counter();fn();sync();elapsed=time.perf_counter()-t
            values.append(elapsed);record(dict(event='sample',seconds=elapsed,repeat=repeat,current_case=current_case))
        return dict(seconds=values,median_seconds=statistics.median(values),minimum_seconds=min(values),
                    maximum_seconds=max(values),cuda_baseline_allocated_bytes=baseline,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if is_cuda else None,
                    cuda_peak_increment_bytes=torch.cuda.max_memory_allocated()-baseline if is_cuda else None)
    try:
        for length in a.lengths:
            deadline();x=torch.randn(a.batch,length,3,a.heads,a.dim,device=a.device,dtype=dtype,requires_grad=True)
            go=torch.randn(a.batch,length,a.heads,a.dim,device=a.device,dtype=dtype)
            methods=dict(legacy_masked_topk=ScheduledAttention(0,'native',a.budget,a.local,a.dropout),
                         chunked_topk=ChunkedTopKAttention(a.budget,a.local,a.dropout,a.query_chunk),
                         dense_sdpa_auto=DenseSDPA(a.dropout),dense_sdpa_flash=DenseSDPA(a.dropout,True))
            for method,model in methods.items():
                for operation in ['prefill_core','attention_forward_backward']:
                    deadline();current_case=dict(length=length,method=method,operation=operation,dtype=a.dtype)
                    row=dict(**current_case,selection_included=method in ['legacy_masked_topk','chunked_topk'],
                             batch=a.batch,heads=a.heads,head_dim=a.dim,
                             dropout_p=a.dropout if operation=='attention_forward_backward' else 0.0)
                    if method=='dense_sdpa_flash' and not is_cuda:
                        row.update(status='not_run',reason='Forced CUDA FlashAttention is not a CPU benchmark');rows.append(row);continue
                    estimate=5*a.batch*a.heads*length*length*x.element_size()
                    if method=='legacy_masked_topk' and estimate>a.max_matrix_gib*1024**3:
                        row.update(status='not_run',reason='Conservative legacy N*N workspace guard',estimate_bytes=estimate);rows.append(row);continue
                    model.train(operation=='attention_forward_backward')
                    def invoke():
                        if operation=='prefill_core':
                            with torch.no_grad():model(x)
                        else:
                            y=model(x);torch.autograd.grad(y,x,go)
                    try:
                        torch.manual_seed(a.seed+length)
                        row.update(status='passed',**samples(invoke))
                        if operation=='attention_forward_backward':row['autograd_storage']=saved_storage(model,x,go)
                        if method.startswith('dense_sdpa'):
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                                invoke();sync()
                            row['observed_attention_operator_names']=sorted({e.key for e in profiler.key_averages() if 'attention' in e.key.lower()})
                    except TimeoutError:raise
                    except Exception as e:
                        row.update(status='unsupported_or_failed',error=str(e),exception=type(e).__name__)
                        if is_cuda:torch.cuda.empty_cache()
                    rows.append(row);record(dict(event='case_complete',result=row))
                    print(json.dumps({k:v for k,v in row.items() if k not in ['autograd_storage','seconds']}),flush=True)
            # Components are diagnostic. Their sum is not substituted for total timings.
            q,k,v=[z.detach().permute(0,2,1,3).reshape(a.batch*a.heads,length,a.dim) for z in x.unbind(2)]
            ids=select_causal_topk(q,k,a.budget,a.local,a.query_chunk)
            for component in ['selector_only','selected_aggregation_only_cached_indices']:
                current_case=dict(length=length,component=component,dtype=a.dtype)
                def part():
                    with torch.no_grad():
                        if component=='selector_only':select_causal_topk(q,k,a.budget,a.local,a.query_chunk)
                        else:selected_attention(q,k,v,ids,query_chunk=a.query_chunk)
                row=dict(**current_case,**samples(part),diagnostic_only=True,
                         selection_included=component=='selector_only')
                components.append(row);record(dict(event='component_complete',result=row))
            del x,go,q,k,v,ids,methods
        status='complete'
    except Exception as e:
        status='incomplete';record(dict(event='failure',error=str(e),traceback=traceback.format_exc()))
    result=dict(status=status,started_utc=started,finished_utc=now(),wall_seconds=time.perf_counter()-tick,
                environment=device_details,rows=rows,components=components,
                analytical_counts={str(n):operation_accounting(n,a.budget,a.local,a.query_chunk,a.batch*a.heads) for n in a.lengths},
                scientific_optimizer_updates=0,cuda_measurements_present=is_cuda and any(r.get('status')=='passed' for r in rows),
                sources_unchanged=all(sha(ROOT/p)==h for p,h in sources.items()),
                limitations=['Exact selection remains quadratic','Unfused PyTorch gather/scatter baseline',
                             'No whole-model training or prompt prefill timing','Random tensors do not establish model quality',
                             'CPU timing and autograd saved storage are not GPU timing or peak VRAM'])
    save(out/'benchmark.json',result)
    if status!='complete' or not result['sources_unchanged']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--dtype',choices=['float32','float64','float16','bfloat16'],default='float32')
    p.add_argument('--lengths',nargs='+',type=int,default=[256,1024]);p.add_argument('--batch',type=int,default=1)
    p.add_argument('--heads',type=int,default=1);p.add_argument('--dim',type=int,default=128)
    p.add_argument('--budget',type=int,default=8);p.add_argument('--local',type=int,default=2)
    p.add_argument('--query-chunk',type=int,default=64);p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=2026091501)
    p.add_argument('--max-seconds',type=float,default=120);p.add_argument('--max-matrix-gib',type=float,default=2)
    main(p.parse_args())
