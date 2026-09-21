"""CUDA numerical checks and bounded technical throughput probe, not research runs."""
import copy,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from zoology_entry import configuration,LanguageModel,set_determinism
from zoology_sparse_schedule import install,set_epoch

def main():
    assert torch.cuda.is_available();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    set_determinism(123);original=LanguageModel(configuration().model).cuda()
    original.backbone.embeddings.device='cuda';changed=copy.deepcopy(original);install(changed,'dense')
    original.eval();changed.eval();x=torch.randint(0,256,(2,64),device='cuda')
    a=original(x);b=changed(x);torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    a.square().mean().backward();b.square().mean().backward()
    for p,q in zip(original.parameters(),changed.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=2e-6)
    install(changed,'native');changed.eval();emb=changed.backbone.embeddings(x[:1]).detach().requires_grad_(True)
    changed.lm_head(changed.backbone.layers_forward(emb))[:,40].square().sum().backward()
    assert float(emb.grad[:,41:].abs().max())==0
    rows=[]
    for method in ['dense','native']:
        set_determinism(123);model=LanguageModel(configuration().model).cuda();model.backbone.embeddings.device='cuda';install(model,method)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001);x=torch.randint(0,256,(32,64),device='cuda');y=torch.randint(0,256,(32,64),device='cuda')
        torch.cuda.reset_peak_memory_stats();seconds=[]
        for step in range(8):
            torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(x).flatten(0,1),y.flatten())
            loss.backward();optimizer.step();torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
            assert torch.isfinite(loss)
        rows.append(dict(method=method,steps=8,mean_last6_seconds=sum(seconds[2:])/6,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
        dense_forward_backward_tolerance_pass=True,future_gradient_zero=True,technical_updates=16,scientific_runs=0,microbenchmarks=rows,
        scope='Dense QK and dense masked AV in both methods; no sparse speedup claim')
    (ROOT/'logs/sparse-schedule-cuda-preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
