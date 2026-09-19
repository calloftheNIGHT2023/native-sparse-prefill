"""All-query prefill timing including summaries, shortlist, fine scores, aggregation.

PyTorch shortlist prototype, NOT a speed claim for Quest/MoBA official kernels.
Random QKV timings do not establish long-context model quality.
"""
import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from block_candidate_topk import BlockCandidateTopKAttention
from triton_topk_selector import TritonRankTopKAttention
from benchmark_chunked_topk import DenseSDPA
from benchmark_topk_cuda_graphs import capture


def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')


def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);start=utc();tick=time.perf_counter();rows=[];checks=[]
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(2026091513)
    sources=['scripts/benchmark_block_candidate_prefill.py','scripts/benchmark_topk_cuda_graphs.py',
        'scripts/benchmark_chunked_topk.py','src/block_candidate_topk.py','src/triton_topk_selector.py','src/triton_selected_attention.py']
    hashes={p:sha(ROOT/p) for p in sources}
    if a.fused_gate:
        gate=json.loads(a.fused_gate.read_text())
        assert gate['status']=='passed' and len(gate['checks'])==30
        assert gate['sources']['src/triton_block_candidate_topk.py']==sha(ROOT/'src/triton_block_candidate_topk.py')
        hashes['src/triton_block_candidate_topk.py']=sha(ROOT/'src/triton_block_candidate_topk.py')
        sources.append('src/triton_block_candidate_topk.py')
    for p in sources:
        dst=out/'source'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/p).read_bytes())
    save(out/'manifest.json',dict(started_utc=start,sources=hashes,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        seed=2026091513,repeats=9,iterations=3,selection_included=True,no_optimizer_updates=True,
        lengths=a.lengths,heads=a.heads,identical_timing_inputs_for_all_methods=True,
        scope='Random BF16 all-query attention core; summaries rebuilt on every call; both graph and eager for every method'))
    def event(x):
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(dict(utc=utc(),elapsed_seconds=time.perf_counter()-tick,**x))+'\n')
        print(json.dumps(x),flush=True)
    def measure(fn):
        left,right=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        wall=time.perf_counter();left.record()
        for _ in range(3):fn()
        right.record();right.synchronize()
        return dict(gpu_ms=left.elapsed_time(right)/3,wall_ms=(time.perf_counter()-wall)*1000/3)
    try:
        for heads in a.heads:
            for n in a.lengths:
                x=torch.randn(1,n,3,heads,128,device='cuda',dtype=torch.bfloat16)
                warmup_input=x.clone();timing_input=torch.randn_like(x)
                methods={'flash':DenseSDPA(0,True).eval(),'exact_top8':TritonRankTopKAttention(dropout=0,query_chunk=1024).eval(),
                    'mean32_r4':BlockCandidateTopKAttention(block=32,routes=4,method='mean',query_chunk=256,backend='triton',dropout=0).eval(),
                    'minmax32_r4':BlockCandidateTopKAttention(block=32,routes=4,method='minmax',query_chunk=256,backend='triton',dropout=0).eval()}
                if a.fused_gate:
                    from triton_block_candidate_topk import FusedBlockCandidateTopKAttention
                    methods={'flash':methods['flash'],'exact_top8':methods['exact_top8'],
                        'fused_mean32_r4':FusedBlockCandidateTopKAttention(method='mean').eval(),
                        'fused_minmax32_r4':FusedBlockCandidateTopKAttention(method='minmax').eval()}
                for name,model in methods.items():
                    ct=time.perf_counter();begin=utc();graph=None;output=None
                    with torch.no_grad():
                        x.copy_(warmup_input)
                        fn=lambda:model(x)
                        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
                        graph,output=capture(fn)
                        capture_seconds=time.perf_counter()-ct;peak=torch.cuda.max_memory_allocated()-base
                        # Move data after capture; output must follow current input, not cached routing.
                        x.copy_(timing_input);expected=fn();graph.replay();torch.cuda.synchronize()
                        torch.testing.assert_close(output,expected,atol=3e-3,rtol=3e-3)
                        checks.append(dict(method=name,length=n,heads=heads,changed_input_graph_matches=True))
                        samples={'eager':[],'graph':[]}
                        for repeat in range(9):
                            for mode in (['eager','graph'] if repeat%2==0 else ['graph','eager']):
                                measured=measure(fn if mode=='eager' else graph.replay);samples[mode].append(measured)
                                event(dict(event='sample',method=name,length=n,heads=heads,repeat=repeat,mode=mode,**measured))
                        row=dict(method=name,length=n,heads=heads,batch=1,dim=128,dtype='bfloat16',operation='prefill_core',
                            started_utc=begin,finished_utc=utc(),wall_seconds=time.perf_counter()-ct,
                            capture_and_warmup_seconds=capture_seconds,capture_peak_increment_bytes=peak,samples=samples,
                            eager_gpu_ms=statistics.median(s['gpu_ms'] for s in samples['eager']),
                            graph_gpu_ms=statistics.median(s['gpu_ms'] for s in samples['graph']))
                        if name=='flash':
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:fn();torch.cuda.synchronize()
                            row['operators']=[e.key for e in prof.key_averages() if 'attention' in e.key]
                        rows.append(row);save(out/'partial.json',rows);event(dict(event='condition_complete',**{k:v for k,v in row.items() if k!='samples'}))
                    del graph,output,expected,model,fn
                del methods,x,warmup_input,timing_input
        status='complete';error=None
    except Exception as exc:
        status='incomplete';error=dict(message=str(exc),traceback=traceback.format_exc());event(dict(event='failed',error=error))
    save(out/'benchmark.json',dict(status=status,error=error,started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,
        conditions=rows,checks=checks,sources=hashes,scientific_optimizer_updates=0,
        protected_files_unchanged=all(sha(ROOT/p)==h for p,h in hashes.items()),
        limitations=['Prototype shortlist, not official optimized MoBA or Quest', 'Independent random inputs, not model quality',
            'All prefill queries and routing work included', 'Capture memory contains graph pool; not persistent KV memory']))
    if status!='complete':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--fused-gate',type=Path)
    p.add_argument('--lengths',type=int,nargs='+',default=[1024,4096,8192]);p.add_argument('--heads',type=int,nargs='+',default=[1,8]);main(p.parse_args())
