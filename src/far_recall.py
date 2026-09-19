"""Known associative recall with counterfactual answers and exact graph checks.

The impossibility bound covers this STATIC sink+recent graph and pointwise
MLPs/residuals only. Content-dependent routing and recurrent mixers are excluded.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from sparse_reference import make_layout,expand_blocks
from routing_rules import route_blocks

def static_support(cfg):
    layout=make_layout([0]*cfg['sequence_length'],cfg['block_size'])
    selected=route_blocks(torch.zeros_like(layout.visible_blocks,dtype=torch.float32),
        layout,cfg['selected_blocks'],'sink_recent')
    return layout,selected,expand_blocks(selected,layout)

def ancestors(mask,query,layers):
    """Backwards transitive support, including residual/pointwise self paths."""
    reach=torch.zeros(len(mask),dtype=torch.bool,device=mask.device); reach[query]=True
    history=[reach.clone()]
    for _ in range(layers):
        reach=reach | mask[reach].any(0); history.append(reach.clone())
    return reach,history

def family(cfg,split,index):
    digest=hashlib.sha256(f"{cfg['seed']}:{split}:{index}".encode()).digest()
    pcg=cfg.get('family_rng')=='numpy_pcg64'
    seed=int.from_bytes(digest if pcg else digest[:8],'big')
    if not pcg: seed%=2**63-1
    # The legacy Torch CPU generator collided for different 64-bit seeds with
    # equal low 32 bits. Preserve old datasets; new stream configs use PCG64.
    if pcg:
        g=np.random.Generator(np.random.PCG64(seed))
        def randint(low,high,shape): return torch.from_numpy(g.integers(low,high,shape,dtype=np.int64))
        def randperm(count): return torch.from_numpy(g.permutation(count))
    else:
        g=torch.Generator().manual_seed(seed)
        def randint(low,high,shape): return torch.randint(low,high,shape,generator=g)
        def randperm(count): return torch.randperm(count,generator=g)
    n=cfg['sequence_length']; records=cfg['records']
    base=randint(cfg['filler_base'],cfg['vocab_size'],(n,))
    slots=torch.arange(cfg['evidence_start'],cfg['evidence_end_exclusive'],4)
    assert len(slots)>=records and slots[-1]+3<n-3
    positions=slots[randperm(len(slots))[:records]]
    keys=cfg['key_base']+randperm(cfg['num_keys'])[:records]
    values=randint(0,cfg['num_values'],(records,))+cfg['value_base']
    target=int(randint(0,records,(1,))); pos=int(positions[target])
    for p,k,v in zip(positions,keys,values):
        base[p:p+4]=torch.tensor([cfg['record_marker'],int(k),int(v),cfg['separator']])
    if cfg.get('query_format')=='key_last':
        base[-3:]=torch.tensor([cfg['question_marker'],cfg['separator'],int(keys[target])])
    else:
        base[-3:]=torch.tensor([cfg['question_marker'],int(keys[target]),cfg['answer_marker']])
    inputs=base.repeat(cfg['num_values'],1)
    labels=torch.arange(cfg['num_values'])
    inputs[:,pos+2]=cfg['value_base']+labels
    meta=dict(id=f'{split}-{index:05d}',seed=seed,target_record=target,value_position=pos+2,
        target_key=int(keys[target]),record_positions=positions.tolist(),query_position=n-1,
        family_sha256=hashlib.sha256(inputs.numpy().tobytes()).hexdigest())
    return inputs,labels,meta

def near_control(inputs,meta,cfg):
    out=inputs.clone(); start=meta['value_position']-2; near=cfg['sequence_length']-cfg.get('near_offset',16)
    assert near>=cfg['evidence_end_exclusive']
    out[:,near:near+4]=inputs[:,start:start+4]
    out[:,start:start+4]=cfg['filler_base']
    return out,near+2

def removed_control(inputs,meta,cfg):
    out=inputs.clone(); out[:,meta['value_position']]=cfg['filler_base']; return out

def oracle_support(layout,selected,meta,cfg):
    """Ground-truth-assisted final-row swap, same block count; not a method."""
    out=selected.clone(); q=meta['query_position']; b=meta['value_position']//cfg['block_size']
    assert layout.visible_blocks[q,b]
    if not out[q,b]:
        old=torch.where(out[q])[0]
        # Preserve first sink and latest complete block when possible.
        removable=old[(old!=old[0]) & (old!=old[-1])]
        drop=int(removable[0] if len(removable) else old[0])
        out[q,drop]=False; out[q,b]=True
    assert torch.equal(out.sum(-1),selected.sum(-1))
    return expand_blocks(out,layout)

def symbolic_lookup(inputs,cfg):
    """Full-context parser upper bound; never reads metadata or target labels."""
    answers=[]
    for x in inputs:
        key=int(x[-1] if cfg.get('query_format')=='key_last' else x[-2]); found=[]
        positions=torch.where((x[:-3]==cfg['record_marker']) & (x[1:-2]==key) & (x[3:]==cfg['separator']))[0]
        for p in positions:
            value=int(x[p+2])-cfg['value_base']
            if 0<=value<cfg['num_values']: found.append(value)
        if len(found)!=1: raise ValueError('Missing/ambiguous answer')
        answers.append(found[0])
    return torch.tensor(answers)

def additive_mask(mask,dtype=torch.float32):
    return torch.zeros_like(mask,dtype=dtype).masked_fill(~mask,torch.finfo(dtype).min)[None,None]
