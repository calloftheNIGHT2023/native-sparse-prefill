"""Frozen-parameter full-model forward/backward timing. ZERO optimizer updates.

All methods use one checkpoint's weights and identical tensors for a compute
comparison. Dense-attention accuracy on these sparse weights is not evaluated.
The trained model length is 256; long-sequence core timings are separate evidence.
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
from router_author_control import make_model
from chunked_topk_attention import ChunkedTopKAttention
from benchmark_chunked_topk import DenseSDPA
from benchmark_topk_cuda_graphs import capture


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.is_available()
    from triton_topk_selector import TritonRankTopKAttention
    for file,source,count in [(a.aggregation_gate,'src/triton_selected_attention.py',30),
                              (a.selector_gate,'src/triton_topk_selector.py',15)]:
        gate=json.loads(file.read_text());assert gate['status']=='passed' and len(gate['checks'])==count
        assert gate['sources'][source]==sha(ROOT/source)
    checkpoint=ROOT/'results/router-author-falsification-lowlr-v0/checkpoint.pt'
    data_path=ROOT/'results/router-author-falsification-evaluation-v0/evaluation-data.pt'
    protected={p:sha(ROOT/p) for p in ['scripts/benchmark_topk_fullmodel.py','scripts/benchmark_topk_cuda_graphs.py',
        'scripts/benchmark_chunked_topk.py','src/chunked_topk_attention.py','src/triton_selected_attention.py',
        'src/triton_topk_selector.py',checkpoint.relative_to(ROOT).as_posix(),data_path.relative_to(ROOT).as_posix()]}
    for path in protected:
        if path.endswith('.py'):
            dst=out/'source'/path;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/path).read_bytes())
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    data=torch.load(data_path,map_location='cpu',weights_only=True)['test']
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4);torch.manual_seed(2026091551)
    start=now();tick=time.perf_counter();rows=[]
    try:
        for dtype_name in a.dtypes:
            for batch in a.batches:
                assert batch<=len(data['inputs'])
                x=data['inputs'][:batch].cuda()
                labels=data['labels'][:batch].cuda()
                # Fixed number of scored tokens; avoid dynamic boolean indexing inside capture.
                positions=torch.where(labels.flatten()!=-100)[0]
                targets=labels.flatten()[positions]
                for method in ['dense_sdpa_auto','dense_sdpa_flash','chunked_topk','triton_rank_topk']:
                    row=dict(method=method,dtype=dtype_name,batch=batch,length=x.shape[1],
                        operation='full_model_forward_backward_without_optimizer',scope='Compute only, not method quality',
                        attention_dropout=.1,selection_included=method in ['chunked_topk','triton_rank_topk'])
                    model=make_model(ckpt['config'],ckpt['model'],'cuda').to(getattr(torch,dtype_name)).train()
                    for layer in model.backbone.layers:
                        inner=(DenseSDPA(.1,method=='dense_sdpa_flash') if method.startswith('dense') else
                            ChunkedTopKAttention(dropout=.1,query_chunk=256) if method=='chunked_topk' else
                            TritonRankTopKAttention(dropout=.1,query_chunk=256))
                        layer.sequence_mixer.inner_attn=inner.train()
                    params=tuple(model.parameters());graph=None;outputs=None
                    def fn():
                        hidden=model.backbone(x).reshape(-1,128).index_select(0,positions)
                        logits=model.lm_head(hidden)
                        loss=torch.nn.functional.cross_entropy(logits.float(),targets)
                        grads=torch.autograd.grad(loss,params)
                        return (loss,*grads)
                    try:
                        torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
                        baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                        begin=time.perf_counter();graph,outputs=capture(fn)
                        row['warmup_capture_seconds']=time.perf_counter()-begin
                        row['capture_peak_increment_bytes']=torch.cuda.max_memory_allocated()-baseline
                        assert all(bool(torch.isfinite(t).all()) for t in outputs)
                        values={'eager':[],'graph':[]}
                        for repeat in range(a.repeats):
                            if time.perf_counter()-tick>a.max_seconds: raise TimeoutError('Wall cap')
                            for mode in (['eager','graph'] if repeat%2==0 else ['graph','eager']):
                                torch.cuda.synchronize()
                                b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                wall=time.perf_counter();b.record()
                                for _ in range(a.iterations): (fn if mode=='eager' else graph.replay)()
                                e.record();e.synchronize()
                                values[mode].append(dict(gpu_ms=b.elapsed_time(e)/a.iterations,
                                    wall_ms=1000*(time.perf_counter()-wall)/a.iterations))
                        row['samples']=values
                        for mode in values:
                            row[mode]={k:statistics.median([v[k] for v in values[mode]]) for k in ['gpu_ms','wall_ms']}
                        row['status']='passed'
                    except TimeoutError: raise
                    except Exception as exc:
                        row.update(status='unsupported_or_failed',error=str(exc),traceback=traceback.format_exc())
                    rows.append(row)
                    with (out/'events.jsonl').open('a') as f: f.write(json.dumps(dict(utc=now(),result=row))+'\n')
                    print(json.dumps({k:v for k,v in row.items() if k not in ['samples','traceback']}),flush=True)
                    del graph,outputs,fn,model,params;gc.collect();torch.cuda.empty_cache()
                del x,labels,positions,targets
        status='complete';error=None
    except Exception as exc:
        status='incomplete';error=dict(message=str(exc),traceback=traceback.format_exc())
    result=dict(status=status,error=error,started_utc=start,finished_utc=now(),wall_seconds=time.perf_counter()-tick,
        rows=rows,sources=protected,protected_files_unchanged=all(sha(ROOT/p)==h for p,h in protected.items()),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),scientific_optimizer_updates=0,
        limitations=['Frozen parameters; optimizer/data transfer excluded','One 1.2M model at its trained length 256',
                     'No new accuracy or learning-curve evidence','Dense method uses same sparse-model weights for timing only',
                     'Mixed precision model quality unverified; prior frozen replay used FP32'])
    (out/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if status!='complete' or not result['protected_files_unchanged']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--aggregation-gate',type=Path,required=True);p.add_argument('--selector-gate',type=Path,required=True)
    p.add_argument('--dtypes',nargs='+',default=['float32','bfloat16']);p.add_argument('--batches',nargs='+',type=int,default=[16,256])
    p.add_argument('--repeats',type=int,default=11);p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--max-seconds',type=float,default=180);main(p.parse_args())
