"""Attention fwd+bwd kernel timing; no model training or optimizer updates."""
import argparse,gc,hashlib,json,random,statistics,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
ROOT=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);start=utc();tick=time.perf_counter();rows=[]
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    def event(kind,**kw):
        row=dict(event=kind,utc=utc(),elapsed_seconds=time.perf_counter()-tick,**kw)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        from flash_moba import flash_moba_varlen_func
        gate=json.loads(a.gate.read_text());assert gate['status']=='passed' and len(gate['checks'])==12
        torch.manual_seed(2026091524);torch.set_num_threads(4)
        save(out/'manifest.json',dict(script_sha256=sha(Path(__file__)),gate_sha256=sha(a.gate),gpu=torch.cuda.get_device_name(),
             torch=torch.__version__,seed=2026091524,lengths=[8192,16384,32768],heads=[1,8],dim=128,batch=1,
             dtype='bfloat16',operation='attention_forward_plus_backward_eager',selection_included=True,repeats=9,calls_per_sample=2,
             scientific_optimizer_updates=0,not_whole_model_training_throughput=True))
        for h in [1,8]:
            for n in [8192,16384,32768]:
                q,k,v=[torch.randn(n,h,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
                go=torch.randn_like(q);cu=torch.tensor([0,n],dtype=torch.int32,device='cuda')
                def dense():
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        return torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True)[0].transpose(0,1)
                methods={'flash_sdpa':dense,'official_moba_b128_k4':lambda:flash_moba_varlen_func(q,k,v,cu,cu,n,n,128,4,causal=True)}
                def invoke(name):return torch.autograd.grad(methods[name](),(q,k,v),go)
                samples={name:[] for name in methods};rng=random.Random(2026091524+n+h)
                for name in methods:
                    ct=time.perf_counter()
                    for _ in range(3):grad=invoke(name)
                    torch.cuda.synchronize();assert all(bool(torch.isfinite(g).all()) for g in grad)
                    event('warmup_complete',method=name,length=n,heads=h,seconds=time.perf_counter()-ct)
                for rep in range(9):
                    order=list(methods);rng.shuffle(order)
                    for name in order:
                        left,right=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        torch.cuda.synchronize();wall=time.perf_counter();left.record()
                        for _ in range(2):grad=invoke(name)
                        right.record();right.synchronize()
                        samples[name].append(dict(repeat=rep,gpu_ms=left.elapsed_time(right)/2,wall_ms=(time.perf_counter()-wall)*1000/2))
                del grad
                for name in methods:
                    gc.collect();torch.cuda.empty_cache();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    grad=invoke(name);torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-base;del grad
                    row=dict(method=name,length=n,heads=h,samples=samples[name],peak_increment_bytes=peak,
                         eager_gpu_ms=statistics.median(s['gpu_ms'] for s in samples[name]),eager_wall_ms=statistics.median(s['wall_ms'] for s in samples[name]))
                    rows.append(row);save(out/'partial.json',rows);event('condition_complete',**{k:v for k,v in row.items() if k!='samples'})
                del q,k,v,go,methods
        status='complete';error=None
    except Exception as exc:status='incomplete';error=dict(message=str(exc),traceback=traceback.format_exc());event('failed',error=error)
    save(out/'benchmark.json',dict(status=status,error=error,conditions=rows,started_utc=start,finished_utc=utc(),
         wall_seconds=time.perf_counter()-tick,scientific_optimizer_updates=0))
    if status!='complete':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);main(p.parse_args())
