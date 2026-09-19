"""CPU forward-only sensitivity probe on the existing 70M/2K cloud checkpoint."""
import hashlib,json,sys,time
from pathlib import Path
import torch
from transformers import AutoConfig,GPTNeoXForCausalLM
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from far_recall import static_support,ancestors,family,near_control,removed_control,oracle_support,additive_mask
from run_joint_pilot import utc,sha,save
def main():
    out=ROOT/'results/far-recall-70m-sensitivity-v1'; out.mkdir(exist_ok=False)
    started=utc(); timer=time.perf_counter(); torch.set_num_threads(4)
    cfg=json.loads((ROOT/'configs/far-recall-v0.json').read_text(encoding='utf-8'))
    folder=ROOT/'results/cloud-dense-context-10m-v0'; manifest=json.loads((folder/'manifest.json').read_text())
    checkpoint=folder/'checkpoint.pt'; expected=next(r['sha256'] for r in manifest if r['path']=='checkpoint.pt')
    assert sha(checkpoint)==expected
    c=AutoConfig.from_pretrained(ROOT/'data/realtext-early-step1000/assets/model',local_files_only=True)
    c._attn_implementation='eager'; model=GPTNeoXForCausalLM(c).float()
    blob=torch.load(checkpoint,map_location='cpu',weights_only=True); model.load_state_dict(blob['model']); del blob
    model.eval(); layout,selected,mask=static_support(cfg); x,y,meta=family(cfg,'test',0)
    near,_=near_control(x,meta,cfg); removed=removed_control(x,meta,cfg)
    oracle=oracle_support(layout,selected,meta,cfg); rows=[]
    for name,inputs,support in [('full_far',x,None),('local_far',x,mask),('local_near',near,mask),
        ('oracle_far',x,oracle),('full_removed',removed,None)]:
        before=time.perf_counter()
        with torch.no_grad():
            # Same batch-1 shape/order for both counterfactuals. The prior
            # batched probe differed by ~3e-6 even for identical inputs.
            outputs=[]
            for example in inputs[:2]:
                hidden=model.gpt_neox(example[None],attention_mask=additive_mask(support) if support is not None else None,
                    use_cache=False).last_hidden_state[:,-1]
                outputs.append(model.embed_out(hidden))
            logits=torch.cat(outputs)
        delta=float((logits[1]-logits[0]).abs().max())
        rows.append(dict(condition=name,forward_examples=2,max_logit_change=delta,seconds=time.perf_counter()-before,
            logits_identical=torch.equal(logits[0],logits[1]),utc=utc()))
        with (out/'events.jsonl').open('a') as f: f.write(json.dumps(rows[-1])+'\n')
        print(json.dumps(rows[-1]),flush=True)
    observed={r['condition']:r for r in rows}
    assert observed['local_far']['logits_identical'] and observed['full_removed']['logits_identical']
    assert all(observed[m]['max_logit_change']>1e-7 for m in ['full_far','local_near','oracle_far'])
    result=dict(started_utc=started,finished_utc=utc(),elapsed_seconds=time.perf_counter()-timer,device='cpu',
        model_parameters=sum(p.numel() for p in model.parameters()),sequence_length=2048,layers=6,
        checkpoint_sha256=expected,checkpoint_read_only=True,conditions=rows,all_checks_passed=True,batch_size=1,
        optimizer_updates=0,gpu_jobs=0,scope='Sensitivity and isolation only; does NOT establish correct answer retrieval or 2K neural learnability')
    save(out/'result.json',result); save(out/'frozen-config.json',cfg)
    save(out/'manifest.json',[dict(path=p.name,sha256=sha(p)) for p in out.iterdir() if p.is_file()])
if __name__=='__main__': main()
