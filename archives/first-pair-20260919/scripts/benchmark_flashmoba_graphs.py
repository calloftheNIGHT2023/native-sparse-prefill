"""Matched eager/graph measurement with fixed metadata and dynamic official routing."""
import argparse,gc,hashlib,json,random,statistics,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flashmoba_fixed_metadata import FlashMoBAFixedMetadata
from benchmark_topk_cuda_graphs import capture
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();start=utc();rows=[]
    sources={p:sha(ROOT/p) for p in ['scripts/benchmark_flashmoba_graphs.py','src/flashmoba_fixed_metadata.py','scripts/benchmark_topk_cuda_graphs.py']}
    for p in sources:
        dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    def event(kind,**kw):
        row=dict(event=kind,utc=utc(),elapsed_seconds=time.perf_counter()-tick,**kw)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        gate=json.loads(a.gate.read_text());assert gate['status']=='passed' and len(gate['checks'])==6
        assert gate['sources']['src/flashmoba_fixed_metadata.py']==sources['src/flashmoba_fixed_metadata.py']
        torch.manual_seed(2026091523);torch.set_num_threads(4)
        save(out/'manifest.json',dict(sources=sources,gate_sha256=sha(a.gate),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            seed=2026091523,causal=True,dtype='bfloat16',dim=128,batch=1,lengths=[8192,16384,32768],heads=[1,8],
            metadata_only_cached=True,all_routing_recomputed=True,scientific_optimizer_updates=0,repeats=11,calls_per_sample=3))
        with torch.no_grad():
            for h in [1,8]:
                for n in [8192,16384,32768]:
                    x=torch.randn(1,n,3,h,128,device='cuda',dtype=torch.bfloat16);q,k,v=[x[0,:,i] for i in range(3)]
                    changed=torch.randn_like(x)
                    def dense():
                        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                            return torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True)
                    methods={'flash_sdpa':dense};models=[]
                    for b,t in [(64,2),(128,2),(128,4)]:
                        m=FlashMoBAFixedMetadata([n],h,128,b,t).eval();models.append(m)
                        methods[f'official_moba_fixedmeta_b{b}_k{t}']=lambda m=m:m(q,k,v)
                    # Capture in turn so graph pools for earlier methods do not pollute allocator statistics.
                    for name,fn in methods.items():
                        begin=utc();t=time.perf_counter();gc.collect();torch.cuda.empty_cache()
                        base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                        graph,output=capture(fn);warm=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated()-base
                        x.copy_(changed);expected=fn();graph.replay();torch.cuda.synchronize()
                        torch.testing.assert_close(output,expected,atol=0,rtol=0)
                        samples={'eager':[],'graph':[]}
                        for repeat in range(11):
                            for mode in (['eager','graph'] if repeat%2==0 else ['graph','eager']):
                                action=fn if mode=='eager' else graph.replay
                                left,right=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                                torch.cuda.synchronize();wall=time.perf_counter();left.record()
                                for _ in range(3):action()
                                right.record();right.synchronize()
                                samples[mode].append(dict(repeat=repeat,gpu_ms=left.elapsed_time(right)/3,wall_ms=(time.perf_counter()-wall)*1000/3))
                        row=dict(method=name,length=n,heads=h,samples=samples,changed_input_graph_equal=True,
                            eager_gpu_ms=statistics.median(s['gpu_ms'] for s in samples['eager']),
                            graph_gpu_ms=statistics.median(s['gpu_ms'] for s in samples['graph']),
                            eager_wall_ms=statistics.median(s['wall_ms'] for s in samples['eager']),
                            graph_wall_ms=statistics.median(s['wall_ms'] for s in samples['graph']),
                            capture_peak_increment_bytes=peak,warmup_capture_seconds=warm,started_utc=begin,finished_utc=utc())
                        rows.append(row);save(out/'partial.json',rows);event('condition_complete',**{k:v for k,v in row.items() if k!='samples'})
                        del graph,output,expected
                    del x,q,k,v,changed,methods,models
        status='complete';error=None
    except Exception as exc:status='incomplete';error=dict(message=str(exc),traceback=traceback.format_exc());event('failed',error=error)
    save(out/'benchmark.json',dict(status=status,error=error,conditions=rows,sources=sources,
        started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,scientific_optimizer_updates=0,
        limitations=['Metadata-only wrapper around official kernels, not unmodified full Python API',
            'Random QKV is not model quality or whole-model speed','Allocation/capture peaks include graph pool']))
    if status!='complete':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);main(p.parse_args())
