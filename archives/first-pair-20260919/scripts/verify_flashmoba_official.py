"""Gate pinned FlashMoBA: independent router, same-support FP32 fwd/bwd, causal perturbation.

No optimizer steps. Hard block selection is differentiated only through its selected Q/K/V.
"""
import argparse, hashlib, json, math, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '39d9ac043b271d046a2181a9991e99a26b67bca1'

def utc(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x): p.write_text(json.dumps(x, indent=2), encoding='utf-8')

def block_masks(q, k, lengths, block, topk):
    """Independent mean routing; current block mandatory, only completed remote blocks."""
    masks=[]; scores=[]; start=0
    for n in lengths:
        qs=q[start:start+n].detach().float().transpose(0,1)
        ks=k[start:start+n].detach()
        means=torch.stack([x.float().mean(0).to(k.dtype) for x in ks.split(block)])
        means=means.repeat_interleave(q.shape[1]//k.shape[1],dim=1).float().transpose(0,1)
        gate=qs @ means.transpose(1,2)
        pos=torch.arange(n,device=q.device)//block
        cols=torch.arange(means.shape[1],device=q.device)
        gate.masked_fill_(cols[None,None,:]>pos[None,:,None],-torch.inf)
        gate.masked_fill_(cols[None,None,:]==pos[None,:,None],torch.inf)
        vals, ids=gate.sort(dim=-1,stable=True)
        ids=ids[:,:,-min(topk,means.shape[1]):];vals=vals[:,:,-ids.shape[-1]:]
        mask=torch.zeros_like(gate,dtype=torch.bool)
        mask.scatter_(-1,ids,vals!=-torch.inf)
        masks.append(mask);scores.append(gate);start+=n
    return masks,scores

def decode_csc(offsets,counts,indices,lengths,heads,block):
    offsets,counts,indices=[x.cpu() for x in (offsets,counts,indices)]
    masks=[]
    for b,n in enumerate(lengths):
        mask=torch.zeros(heads,n,math.ceil(n/block),dtype=torch.bool)
        for h in range(heads):
            for c in range(mask.shape[-1]):
                lo=int(offsets[b,h,c]);num=int(counts[b,h,c]);ids=indices[lo:lo+num].long()
                assert bool(((ids>=0)&(ids<n)).all()) and len(ids.unique())==num
                mask[h,ids,c]=True
        masks.append(mask.to('cuda'))
    return masks

def reference(q,k,v,lengths,masks,block):
    outputs=[];start=0;dim=q.shape[-1]
    for n,mask in zip(lengths,masks):
        qs=q[start:start+n].transpose(0,1)
        ks=k[start:start+n].repeat_interleave(q.shape[1]//k.shape[1],1).transpose(0,1)
        vs=v[start:start+n].repeat_interleave(q.shape[1]//v.shape[1],1).transpose(0,1)
        pos=torch.arange(n,device=q.device)
        tokenmask=mask[:,:,pos//block] & (pos[None,None,:]<=pos[None,:,None])
        outputs.append((((qs@ks.transpose(1,2))/math.sqrt(dim)).masked_fill(~tokenmask,-torch.inf).softmax(-1)@vs).transpose(0,1))
        start+=n
    return torch.cat(outputs)

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();start=utc();rows=[]
    sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    (out/'source').mkdir();(out/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    def event(kind,**kw):
        row=dict(utc=utc(),elapsed_seconds=time.perf_counter()-tick,event=kind,**kw)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        from flash_moba import flash_moba_varlen_func,flash_topk_varlen_func
        import flash_moba, flash_moba_cuda
        repo=ROOT/'third_party/flash-moba-official-20260915'
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==COMMIT
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
        torch.set_num_threads(4);torch.manual_seed(2026091520);torch.backends.cuda.matmul.allow_tf32=False
        save(out/'manifest.json',dict(commit=COMMIT,sources=sources,torch=torch.__version__,cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(),module=flash_moba.__file__,extension_sha256=sha(Path(flash_moba_cuda.__file__)),
             seed=2026091520,scientific_optimizer_updates=0,planned_cases=12))
        cases=[([257],1,1,64,64,2),([513],4,4,128,128,2),([129,517],4,2,128,64,4),
               ([256],2,2,64,64,4),([65,193],4,1,64,64,2),([1024],8,8,128,128,4)]
        for dtype in [torch.float16,torch.bfloat16]:
            for lengths,h,hk,d,block,topk in cases:
                q=torch.randn(sum(lengths),h,d,device='cuda',dtype=dtype,requires_grad=True)
                k=torch.randn(sum(lengths),hk,d,device='cuda',dtype=dtype,requires_grad=True);v=torch.randn_like(k,requires_grad=True)
                cu=torch.tensor([0]+list(torch.tensor(lengths).cumsum(0).tolist()),device='cuda',dtype=torch.int32)
                args=(cu,cu,max(lengths),max(lengths))
                with torch.no_grad():
                    offsets,counts,indices=flash_topk_varlen_func(q,k,*args,topk,block,causal=True)
                    masks=decode_csc(offsets,counts,indices,lengths,h,block)
                    independent,scores=block_masks(q,k,lengths,block,topk)
                    disagreements=0;max_regret=0.
                    for mask,ref,gate,n in zip(masks,independent,scores,lengths):
                        pos=torch.arange(n,device='cuda')//block
                        assert bool(mask[:,torch.arange(n,device='cuda'),pos].all())
                        assert not bool((mask & (torch.arange(mask.shape[-1],device='cuda')[None,None,:]>pos[None,:,None])).any())
                        assert torch.equal(mask.sum(-1),ref.sum(-1))
                        disagree=(mask!=ref).any(-1);disagreements+=int(disagree.sum())
                        # Exact ties / floating reduction ordering may alter only almost-equal gate scores.
                        current=torch.arange(mask.shape[-1],device='cuda')[None,None,:]==pos[None,:,None]
                        finite=gate.masked_fill(current,0).masked_fill(gate==-torch.inf,0)
                        regret=(finite*ref).sum(-1)-(finite*mask).sum(-1)
                        max_regret=max(max_regret,float(regret.max()))
                        assert float(regret.max())<2e-4
                actual=flash_moba_varlen_func(q,k,v,*args,block,topk,causal=True)
                qr,kr,vr=[x.detach().float().requires_grad_() for x in (q,k,v)]
                expected=reference(qr,kr,vr,lengths,masks,block)
                go=torch.randn_like(actual)
                ga=torch.autograd.grad(actual,(q,k,v),go);gr=torch.autograd.grad(expected,(qr,kr,vr),go.float())
                tol=.025 if dtype==torch.bfloat16 else .004
                torch.testing.assert_close(actual.float(),expected,atol=tol,rtol=tol)
                for x,y in zip(ga,gr):torch.testing.assert_close(x.float(),y,atol=tol,rtol=tol)
                with torch.no_grad():
                    kp,vp=k.clone(),v.clone();st=0
                    for n in lengths:
                        kp[st+n//2:st+n]=torch.randn_like(kp[st+n//2:st+n])*5
                        vp[st+n//2:st+n]=torch.randn_like(vp[st+n//2:st+n])*5;st+=n
                    changed=flash_moba_varlen_func(q,kp,vp,*args,block,topk,causal=True);st=0
                    for n in lengths:
                        torch.testing.assert_close(actual[st:st+n//2],changed[st:st+n//2],atol=0,rtol=0);st+=n
                row=dict(dtype=str(dtype),lengths=lengths,heads=h,kv_heads=hk,dim=d,block=block,topk_including_current=topk,
                    routing_disagreement_rows=disagreements,maximum_gate_score_regret=max_regret,
                    output_max_abs=float((actual.float()-expected).abs().max()),
                    gradient_max_abs=max(float((x.float()-y).abs().max()) for x,y in zip(ga,gr)),
                    future_perturbation_prefix_unchanged=True,status='passed')
                rows.append(row);event('case_complete',**row);save(out/'partial.json',rows)
        status='passed';error=None
    except Exception as exc:
        status='failed';error=dict(message=str(exc),traceback=traceback.format_exc());event('failed',error=error)
    save(out/'verification.json',dict(status=status,error=error,started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,
         checks=rows,sources=sources,commit=COMMIT,scientific_optimizer_updates=0))
    if status!='passed':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
