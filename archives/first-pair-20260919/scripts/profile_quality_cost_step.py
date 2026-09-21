"""One instrumented forward/backward at identical initialization; zero optimizer updates."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from run_pool_training_control import LoRALinear
import run_flashmoba_realtext_precision as base
ROOT=Path(__file__).resolve().parents[1]

def main(a):
    out=ROOT/'results'/f'flashmoba-quality-cost-profile-{a.condition}-v0';out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'data/flashmoba-quality-cost-v0/config.json').read_text());cond=next(c for c in cfg['conditions'] if c['name']==a.condition)
    assert hashlib.sha256(Path(base.flash_moba_cuda.__file__).read_bytes()).hexdigest()==cfg['extension_sha256'][cond['extension']]
    start=datetime.now(timezone.utc).isoformat();torch.set_num_threads(4);torch.manual_seed(cfg['initialization_seeds'][0]);torch.backends.cuda.matmul.allow_tf32=False
    base.LOCKED_POOL_CONFIG=tuple(cfg['pool_config'])
    model=AutoModelForCausalLM.from_pretrained(ROOT/cfg['model_path'],local_files_only=True,trust_remote_code=False,
        use_safetensors=True,attn_implementation='eager',dtype=torch.float32).cuda().eval();model.requires_grad_(False)
    torch.manual_seed(cfg['initialization_seeds'][0])
    for layer in model.model.layers:
        for name in ['q_proj','k_proj','v_proj','o_proj']:
            setattr(layer.self_attn,name,LoRALinear(getattr(layer.self_attn,name),8,16))
    weights={n:p for n,p in model.named_parameters() if p.requires_grad}
    ih=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in weights.values())).hexdigest()
    expected=json.loads((ROOT/'results'/f"flashmoba-quality-cost-{a.condition}-s{cfg['initialization_seeds'][0]}-v1/manifest.json").read_text())
    assert ih==expected['initial_adapters_tensor_sha256']
    def attention(module,query,key,value,attention_mask,scaling=None,dropout=0.,**kwargs):
        q,k,v=[z[0].transpose(0,1).to(torch.bfloat16).contiguous() for z in (query,key,value)]
        with torch.profiler.record_function('attention_adapter_'+cond['pool']):
            if cond['pool']=='dense':
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    y=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True,dropout_p=0.,scale=scaling,enable_gqa=True)
                return y.transpose(1,2).to(query.dtype),None
            y=base.sparse(q,k,v,128,cond['topk'],cond['pool'],scaling)
            return y[None].to(query.dtype),None
    ALL_ATTENTION_FUNCTIONS.register('quality_cost_profile',attention);model.config._attn_implementation='quality_cost_profile'
    data=np.load(ROOT/'data/flashmoba-quality-cost-v0/tokens.npz')['train'][0]
    ids=torch.tensor(data[:-1][None],device='cuda');target=torch.tensor(data[1:],device='cuda')
    def step():
        model.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(ids,use_cache=False).logits[0],target);loss.backward();return loss
    for _ in range(3):loss=step()
    torch.cuda.synchronize();wall=time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        loss=step();torch.cuda.synchronize()
    measured_wall=time.perf_counter()-wall
    prof.export_chrome_trace(str(out/'trace.json'))
    averages=[]
    for e in prof.key_averages():
        averages.append(dict(name=e.key,calls=e.count,cpu_self_us=e.self_cpu_time_total,device_self_us=e.self_device_time_total,device_total_us=e.device_time_total))
    averages.sort(key=lambda x:x['device_self_us'],reverse=True)
    trace=json.loads((out/'trace.json').read_text());groups={};kernels=[]
    for e in trace['traceEvents']:
        if e.get('cat')!='kernel' or e.get('ph')!='X':continue
        name=e['name'];duration=e.get('dur',0.)
        lower=name.lower()
        if 'flash_moba' in lower or 'pytorch_flash' in lower or 'flash_fwd' in lower or 'flash_bwd' in lower:group='attention_kernels'
        elif 'mean_pool' in lower:group='pooling'
        elif 'topk' in lower or 'sort' in lower or 'cub::' in lower:group='routing_sorting_and_cub'
        elif 'gemm' in lower or 'gemv' in lower:group='other_matrix_kernels'
        else:group='other_kernels'
        groups[group]=groups.get(group,0.)+duration;kernels.append(dict(name=name,microseconds=duration,group=group))
    result=dict(status='complete',condition=a.condition,started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),
        scientific_optimizer_updates=0,instrumented_forward_backward_wall_seconds=measured_wall,loss=float(loss.detach()),
        grouped_cuda_kernel_microseconds=groups,cuda_kernel_microseconds_total=sum(groups.values()),cuda_kernel_events=len(kernels),
        averages=averages,kernels=kernels,initial_adapters_sha256=ih,
        classification='Name-based heuristic; inspect raw trace to refine. Sum of kernel durations is not critical-path wall time. Instrumented run is not the timing benchmark, excludes optimizer and input transfer, uses initial adapters.',
        extension_sha256=hashlib.sha256(Path(base.flash_moba_cuda.__file__).read_bytes()).hexdigest())
    (out/'source.py').write_bytes(Path(__file__).read_bytes());(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['averages','kernels']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);main(p.parse_args())
