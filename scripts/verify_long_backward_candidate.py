"""Cloud-ready gate: actual K4 metadata, same-graph repeats, and offline FP64 reference.

Run once with the original extension and once in a fresh process with an isolated
candidate build on PYTHONPATH. This does not install/replace the original extension.
"""
import argparse,hashlib,json,math,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import torch
import flash_moba_cuda
from flash_moba.flash_moba_interface import flash_moba_attn_varlen_func,decide_lg_block_m
ROOT=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();start=utc();rows=[]
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    def event(kind,**kw):
        obj=dict(utc=utc(),event=kind,elapsed_seconds=time.perf_counter()-tick,**kw)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(obj)+'\n')
        print(json.dumps(obj),flush=True)
    try:
        torch.set_num_threads(4);torch.manual_seed(2026091547);torch.backends.cuda.matmul.allow_tf32=False
        qpath=a.bundle/'qkv.pt'
        z=torch.load(qpath,weights_only=True);q,k,v=[z[n].cuda().requires_grad_() for n in ['q','k','v']]
        go=torch.load(a.bundle/'upstream-gradient.pt',weights_only=True).cuda()
        ref=torch.load(a.bundle/'fp64-reference.pt',weights_only=True)
        reconstructed=torch.load(a.bundle/'reconstructed-routes.pt',weights_only=True)['routes']
        # Use actual saved FP32-pooled-then-BF16 keys, identical for the audited kernel configs.
        means=torch.load(a.bundle/'pooled-means.pt',weights_only=True).cuda()
        n,h,d=q.shape;b=128;topk=4;cu=torch.tensor([0,n],device='cuda',dtype=torch.int32);cm=torch.tensor([0,n//b],device='cuda',dtype=torch.int32)
        offsets,counts,indices,_,ids=flash_moba_cuda.moba_fused_topk(q,means,cu,cu,cm,n,n,topk,b,True)
        indices=flash_moba_cuda.varlen_sort(offsets.flatten(),(offsets+counts).flatten(),indices)
        ids=ids[...,:topk].sort(-1).values.cpu();match=(ids.long()==reconstructed.sort(-1).values).all(-1)
        pos,head=ref['query_positions'],ref['query_heads'];selected_match=match[pos,head]
        torch.save(dict(ids=ids,offsets=offsets.cpu(),counts=counts.cpu(),indices=indices.cpu()),out/'actual-routing.pt')
        event('actual_route_check',all_rows=match.numel(),changed_rows=int((~match).sum()),selected_rows=len(pos),selected_changed_rows=int((~selected_match).sum()))
        assert bool(selected_match.all()),'CPU FP64 reference uses a different selected mask; do not compare gradients as if masks matched.'
        saved=[]
        for deterministic in [False,True]:
            output=flash_moba_attn_varlen_func(q,k,v,cu,cu,n,n,offsets,counts,indices,decide_lg_block_m(topk,b,n,True),b,dropout_p=0.,causal=True,deterministic=deterministic)
            baseline=None
            for repeat in range(a.repeats):
                grads=torch.autograd.grad(output,(q,k,v),go,retain_graph=True)
                if baseline is None:baseline=[g.clone() for g in grads]
                checks=[]
                for old,new in zip(baseline,grads):
                    delta=new.float()-old.float();checks.append(dict(relative_l2=float(delta.norm()/old.float().norm().clamp_min(1e-30)),max_abs=float(delta.abs().max()),changed_elements=int((old!=new).sum())))
                selected=grads[0][pos,head].cpu().double();diff=selected-ref['reference_q_gradients']
                row=dict(deterministic=deterministic,repeat=repeat,qkv_repeat=checks,reference_relative_l2=float(diff.norm()/ref['reference_q_gradients'].norm()),reference_max_abs=float(diff.abs().max()))
                rows.append(row);saved.append(dict(deterministic=deterministic,repeat=repeat,selected_q_gradients=selected));event('backward_complete',**row)
            del output,baseline,grads
        torch.save(saved,out/'selected-gradients.pt')
        deterministic_repeat_pass=all(x['qkv_repeat'][0]['changed_elements']==0 and x['qkv_repeat'][1]['changed_elements']==0 and x['qkv_repeat'][2]['changed_elements']==0 for x in rows if x['deterministic'])
        reference_tolerance_pass=all(x['reference_relative_l2']<=.05 for x in rows)
        result=dict(status='complete',label=a.label,started_utc=start,finished_utc=utc(),seconds=time.perf_counter()-tick,
            actual_k4_selected_masks_match=True,actual_k4_all_changed_rows=int((~match).sum()),repeats_per_mode=a.repeats,rows=rows,
            extension_path=flash_moba_cuda.__file__,extension_sha256=sha(Path(flash_moba_cuda.__file__)),source_sha256=sha(Path(__file__)),input_sha256=sha(qpath),
            gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),deterministic_repeat_pass=deterministic_repeat_pass,
            selected_reference_relative_tolerance=.05,reference_tolerance_pass=reference_tolerance_pass,scientific_optimizer_updates=0,
            scope='Reference rows selected partly for large prior repeat differences, not unbiased full-tensor accuracy. Patch candidate only passes if masks match, all reference comparisons pass, deterministic repeats match, and separate standard gates pass. No training speed/quality/novelty claim.')
    except Exception:
        result=dict(status='failed',label=a.label,started_utc=start,finished_utc=utc(),rows=rows,error=traceback.format_exc(),scientific_optimizer_updates=0)
        event('failed',error=result['error'])
    (out/'result.json').write_text(json.dumps(result,indent=2))
    if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--repeats',type=int,default=16)
    p.add_argument('--bundle',type=Path,default=ROOT/'data/flashmoba-backward-fixed-v0');main(p.parse_args())
