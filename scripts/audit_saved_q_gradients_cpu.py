"""Offline FP64 Q-gradient diagnostic; reconstructed K4 routes are explicitly provisional."""
import hashlib,json,math,time
from pathlib import Path
from datetime import datetime,timezone
import torch
ROOT=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=ROOT/'results/flashmoba-backward-cpu-reference-v1';out.mkdir(parents=True,exist_ok=False)
start=utc();tick=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(2026091547)
(out/'source.py').write_bytes(Path(__file__).read_bytes())
qp=ROOT/'results/flashmoba-qwen-long-precision-v0/fixed-real-qkv-example.pt'
mp=ROOT/'results/flashmoba-pool-autotune-v0/pooled-means-by-config.pt'
gp=ROOT/'results/flashmoba-fixed-graph-backward-v0/gradients.pt'
op=ROOT/'results/flashmoba-pool-repair-v1/gradient-repeats.pt'
z=torch.load(qp,weights_only=True);q,k,v=[z[n] for n in ['q','k','v']]
means=torch.load(mp,weights_only=True)['torch.float32_bn32_w4_s3']
grads=torch.load(gp,weights_only=True)['gradients'];go=torch.load(op,weights_only=True)['upstream_gradient']
n,h,d=q.shape;block=128;topk=4;nb=n//block;ratio=h//k.shape[1]
delta=grads[1][0].float()-grads[0][0].float();changed=delta.square().sum(-1)
ranked=changed.flatten().argsort(descending=True)[:64]
random=torch.randperm(n*h)[:64]
selected=torch.unique(torch.cat([ranked,random]),sorted=True)
positions=selected//h;heads=selected%h;types=['large_repeat_difference' if int(x) in ranked.tolist() else 'random_control' for x in selected]
routes=torch.full((n,h,topk),-1,dtype=torch.int64);route2=torch.full((n,h,2),-1,dtype=torch.int64)
scores_saved=[];margins=torch.zeros((n,h),dtype=torch.float64);bound=torch.zeros_like(margins)
# FP64 products of stored BF16 input values, with actual saved pooled BF16 keys.
# K2 is independently compared with the stored GPU IDs. K4 GPU IDs were not saved.
for head in range(h):
    scores=q[:,head].double()@means[:,head//ratio].double().T
    col=torch.arange(nb)[None];curr=torch.arange(n)[:,None]//block
    remote=scores.masked_fill(col>=curr,-torch.inf)
    vals,ids=remote.sort(dim=-1,descending=True,stable=True)
    routes[:,head,0]=curr[:,0];routes[:,head,1:]=ids[:,:3].masked_fill(~torch.isfinite(vals[:,:3]),-1)
    route2[:,head,0]=curr[:,0];route2[:,head,1]=ids[:,0].masked_fill(~torch.isfinite(vals[:,0]),-1)
    margins[:,head]=vals[:,2]-vals[:,3]
    absdot=q[:,head].double().abs()@means[:,head//ratio].double().abs().T
    gamma=(63*2**-24)/(1-63*2**-24)
    bound[:,head]=(gamma*absdot).masked_fill(col>=curr,0).max(-1).values
saved2=z['fp32_pool_ids'].long().sort(-1).values
comparison2=(route2.sort(-1).values!=saved2).any(-1)
torch.save(dict(routes=routes,selection=selected,positions=positions,heads=heads,scope='K4 reconstructed from saved pooled keys; must check against actual GPU metadata before making an accuracy claim.'),out/'reconstructed-routes.pt')
references=[];autograd_max=0.;records=[]
for i,(pos,head) in enumerate(zip(positions.tolist(),heads.tolist())):
    block_ids=routes[pos,head];block_ids=block_ids[block_ids>=0]
    token_ids=(block_ids[:,None]*block+torch.arange(block)[None]).flatten();token_ids=token_ids[token_ids<=pos]
    assert len(token_ids.unique())==len(token_ids) and pos in token_ids
    keys=k[token_ids,head//ratio].double();values=v[token_ids,head//ratio].double()
    qq=q[pos,head].double().requires_grad_();gg=go[pos,head].double()
    prob=(qq@keys.T/math.sqrt(d)).softmax(-1);output=prob@values
    auto=torch.autograd.grad((output*gg).sum(),qq)[0]
    with torch.no_grad():
        dprob=values@gg;ds=prob*(dprob-(prob*dprob).sum());analytic=ds@keys/math.sqrt(d)
    torch.testing.assert_close(analytic,auto,rtol=1e-10,atol=1e-10)
    autograd_max=max(autograd_max,float((analytic-auto).abs().max()))
    references.append(analytic)
    records.append(dict(position=pos,head=head,selection=types[i],blocks=block_ids.tolist(),valid_keys=len(token_ids),
        reconstructed_remote_cutoff_margin=float(margins[pos,head]) if torch.isfinite(margins[pos,head]) else None,standard_fp32_dot_error_bound=float(bound[pos,head]),
        stable_under_standard_dot_error_model=pos//block<=3 or bool(margins[pos,head]>2*bound[pos,head]),
        saved_repeat_difference_l2=float(delta[pos,head].norm())))
ref=torch.stack(references);torch.save(dict(reference_q_gradients=ref,query_positions=positions,query_heads=heads),out/'fp64-reference.pt')
comparisons=[]
for repeat,g in enumerate(grads):
    actual=g[0][positions,heads].double();diff=actual-ref
    comparisons.append(dict(repeat=repeat,relative_l2=float(diff.norm()/ref.norm()),max_abs=float(diff.abs().max()),
        per_query_relative_l2=(diff.norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-12)).tolist()))
result=dict(status='complete_provisional_route_reference',started_utc=start,finished_utc=utc(),seconds=time.perf_counter()-tick,
    source_sha256=sha(Path(__file__)),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [qp,mp,gp,op]},
    saved_k2_route_rows=n*h,reconstructed_k2_changed_rows=int(comparison2.sum()),k4_gpu_routes_saved=False,
    selected_query_head_rows=len(records),fp64_analytic_vs_autograd_max_abs=autograd_max,records=records,comparisons=comparisons,
    scientific_optimizer_updates=0,new_gpu_calls=0,
    scope='FP64 reference exact for reconstructed fixed K4 mask. K2 matches can support router reconstruction but cannot replace direct K4 metadata verification. Comparisons to old GPU gradients remain provisional until K4 sets match. No root-cause or fix claim.')
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['records','comparisons','input_sha256']},indent=2))
print(json.dumps([dict(repeat=x['repeat'],relative_l2=x['relative_l2'],max_abs=x['max_abs']) for x in comparisons],indent=2))
