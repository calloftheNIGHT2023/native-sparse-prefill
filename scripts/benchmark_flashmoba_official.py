"""Pinned official FlashMoBA versus forced PyTorch Flash SDPA, matched random inputs.

Primary end-to-end eager timing includes mean pooling, routing, sorting, aggregation.
No graph timing: upstream router materializes a sequence-metadata scalar on host.
Block-attention and token-topk outputs are NOT treated as equivalent algorithms.
"""
import argparse,gc,hashlib,json,random,statistics,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);start=utc();tick=time.perf_counter();rows=[]
    source=sha(Path(__file__));(out/'source.py').write_bytes(Path(__file__).read_bytes())
    def event(kind,**kw):
        row=dict(utc=utc(),elapsed_seconds=time.perf_counter()-tick,event=kind,**kw)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        from flash_moba import flash_moba_varlen_func,flash_topk_varlen_func,flash_moba_attn_varlen_func
        from flash_moba.flash_moba_interface import decide_lg_block_m
        gate=json.loads(a.gate.read_text());assert gate['status']=='passed' and len(gate['checks'])==12
        assert gate['sources']['scripts/verify_flashmoba_official_v1.py']==sha(ROOT/'scripts/verify_flashmoba_official_v1.py')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'third_party/flash-moba-official-20260915',text=True).strip()
        assert commit==gate['commit']
        torch.set_num_threads(4);torch.manual_seed(2026091521);torch.backends.cuda.matmul.allow_tf32=False
        save(out/'manifest.json',dict(started_utc=start,script_sha256=source,gate_sha256=sha(a.gate),commit=commit,
            torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,seed=2026091521,
            lengths=a.lengths,heads=a.heads,repeats=11,calls_per_sample=3,causal=True,dtype='bfloat16',batch=1,dim=128,
            selection_included=True,timing='eager end-to-end CUDA events and synchronized wall clock',
            scientific_optimizer_updates=0,random_qkv_not_model_quality=True))
        for h in a.heads:
            for n in a.lengths:
                x=torch.randn(1,n,3,h,128,device='cuda',dtype=torch.bfloat16)
                q,k,v=[x[0,:,i] for i in range(3)]
                cu=torch.tensor([0,n],dtype=torch.int32,device='cuda')
                args=(cu,cu,n,n)
                def dense():
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        return torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True)
                methods={'flash_sdpa':dense}
                configs={'flash_sdpa':dict(selection='all causal tokens')}
                for block,topk in [(64,2),(128,2),(128,4)]:
                    name=f'official_moba_b{block}_k{topk}'
                    methods[name]=lambda b=block,t=topk:flash_moba_varlen_func(q,k,v,*args,b,t,causal=True)
                    configs[name]=dict(block=block,topk_including_current=topk,maximum_tokens_per_query=block*topk)
                if a.include_prototype:
                    from triton_block_candidate_topk import FusedBlockCandidateTopKAttention
                    proto=FusedBlockCandidateTopKAttention(method='minmax').eval()
                    methods['prototype_minmax_b32_r4_token8']=lambda:proto(x)
                    configs['prototype_minmax_b32_r4_token8']=dict(block=32,remote_blocks=4,final_tokens=8,
                        warning='different attention algorithm; not a quality-matched speed comparison')
                with torch.no_grad():
                    # Compile/autotune each method before all interleaved timing samples.
                    for name,fn in methods.items():
                        began=time.perf_counter();y=fn()
                        for _ in range(5):y=fn()
                        torch.cuda.synchronize();assert bool(torch.isfinite(y).all())
                        event('warmup_complete',method=name,length=n,heads=h,seconds=time.perf_counter()-began)
                    samples={name:[] for name in methods};rng=random.Random(2026091521+n+h)
                    for repeat in range(11):
                        order=list(methods);rng.shuffle(order)
                        for name in order:
                            fn=methods[name];left,right=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                            torch.cuda.synchronize();wall=time.perf_counter();left.record()
                            for _ in range(3):y=fn()
                            right.record();right.synchronize()
                            sample=dict(repeat=repeat,gpu_ms=left.elapsed_time(right)/3,wall_ms=(time.perf_counter()-wall)*1000/3)
                            samples[name].append(sample)
                    for name,fn in methods.items():
                        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                        y=fn();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-base
                        row=dict(method=name,length=n,heads=h,**configs[name],samples=samples[name],
                            eager_gpu_ms=statistics.median(s['gpu_ms'] for s in samples[name]),
                            eager_wall_ms=statistics.median(s['wall_ms'] for s in samples[name]),
                            peak_increment_bytes=peak)
                        if name=='flash_sdpa':
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:fn();torch.cuda.synchronize()
                            row['operators']=[e.key for e in prof.key_averages() if 'attention' in e.key]
                        rows.append(row);save(out/'partial.json',rows);event('condition_complete',**{k:v for k,v in row.items() if k!='samples'})
                del y,methods,x,q,k,v
        status='complete';error=None
    except Exception as exc:
        status='incomplete';error=dict(message=str(exc),traceback=traceback.format_exc());event('failed',error=error)
    save(out/'benchmark.json',dict(status=status,error=error,started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,
        conditions=rows,scientific_optimizer_updates=0,script_sha256=source,
        limitations=['Random QKV is not model quality','Eager timing includes official host synchronization',
        'No graph comparison with cached routes','Different topk token and block algorithms are not quality matched']))
    if status!='complete':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',default=[8192,16384,32768]);p.add_argument('--heads',type=int,nargs='+',default=[1,8])
    p.add_argument('--include-prototype',action='store_true');main(p.parse_args())
