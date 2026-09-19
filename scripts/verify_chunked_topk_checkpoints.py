"""Zero-update implementation replay of already-exposed evaluation examples."""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from router_author_control import make_model, selected_logits
from zoology_sparse_schedule import install
from chunked_topk_attention import install_chunked


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main(a):
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=ROOT/'results/router-author-falsification-evaluation-v0'
    old=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    checkpoints=['results/router-author-falsification-lowlr-v0/checkpoint.pt',
                 'results/router-author-falsification-confirm-v0/checkpoint.pt']
    references=[next(row for row in old['conditions'] if row['checkpoint']==p) for p in checkpoints]
    protected={str(p.relative_to(ROOT)):sha(p) for p in [folder/'evaluation.json',folder/'evaluation-data.pt',
               *[ROOT/p for p in checkpoints],ROOT/'src/chunked_topk_attention.py',
               ROOT/'src/zoology_sparse_schedule.py',Path(__file__)]}
    if a.backend != 'chunked':
        assert a.triton_gate is not None
        gate=json.loads(a.triton_gate.read_text(encoding='utf-8'))
        assert gate['status']=='passed' and len(gate['checks'])==30
        assert gate['sources']['src/triton_selected_attention.py']==sha(ROOT/'src/triton_selected_attention.py')
        protected['src/triton_selected_attention.py']=sha(ROOT/'src/triton_selected_attention.py')
        if a.backend=='triton_rank':
            assert a.selector_gate is not None
            sg=json.loads(a.selector_gate.read_text(encoding='utf-8'))
            assert sg['status']=='passed' and len(sg['checks'])==15
            assert sg['sources']['src/triton_topk_selector.py']==sha(ROOT/'src/triton_topk_selector.py')
            protected['src/triton_topk_selector.py']=sha(ROOT/'src/triton_topk_selector.py')
    save(out/'frozen-inputs.json',dict(utc=now(),files=protected,rows=a.rows or 'all',dtype=a.dtype,logit_policy=a.logit_policy,
         backend=a.backend,role='Previously exposed examples for implementation equivalence; no new generalization test'))
    for rel in ['src/chunked_topk_attention.py','src/zoology_sparse_schedule.py','scripts/verify_chunked_topk_checkpoints.py']:
        dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    for rel in protected:
        if rel.startswith('src/triton'):
            dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    data=torch.load(folder/'evaluation-data.pt',map_location='cpu',weights_only=True)
    started=now();tick=time.perf_counter();checks=[];total=0
    try:
        for ref in references:
            path=ROOT/ref['checkpoint'];assert sha(path)==ref['checkpoint_sha256']
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            old_model=make_model(checkpoint['config'],checkpoint['model'],a.device)
            new_model=make_model(checkpoint['config'],checkpoint['model'],a.device)
            old_model.to(dtype=getattr(torch,a.dtype));new_model.to(dtype=getattr(torch,a.dtype))
            install(old_model,'native');install_chunked(new_model,query_chunk=a.query_chunk)
            if a.backend != 'chunked':
                from triton_selected_attention import TritonTopKAttention
                cls=TritonTopKAttention
                if a.backend=='triton_rank':
                    from triton_topk_selector import TritonRankTopKAttention
                    cls=TritonRankTopKAttention
                for layer in new_model.backbone.layers:
                    layer.sequence_mixer.inner_attn=cls(query_chunk=a.query_chunk)
            old_model.eval();new_model.eval()
            for split in ['test','swapped','noisy']:
                part=data[split];rows=len(part['inputs']) if not a.rows else min(a.rows,len(part['inputs']))
                preds=[];max_error=0.0;matching_logits=True;correct=0;count=0;bad_entries=0;old_nll=0.;new_nll=0.
                with torch.no_grad():
                    for first in range(0,rows,a.batch):
                        if time.perf_counter()-tick>a.max_seconds:raise TimeoutError('Replay wall-time cap')
                        x=part['inputs'][first:min(first+a.batch,rows)].to(a.device)
                        y=part['labels'][first:min(first+a.batch,rows)].to(a.device)
                        lo,targets=selected_logits(old_model,x,y);ln,targets2=selected_logits(new_model,x,y)
                        assert torch.equal(targets,targets2)
                        max_error=max(max_error,float((lo-ln).abs().max()))
                        matching_logits=matching_logits and bool(torch.allclose(lo,ln,rtol=1e-4,atol=1e-4))
                        bad_entries+=int((~torch.isclose(lo,ln,rtol=1e-4,atol=1e-4)).sum())
                        old_nll+=float(torch.nn.functional.cross_entropy(lo,targets,reduction='sum'))
                        new_nll+=float(torch.nn.functional.cross_entropy(ln,targets,reduction='sum'))
                        po,pn=lo.argmax(-1),ln.argmax(-1)
                        assert torch.equal(po,pn),'Changed predicted answers'
                        batch_preds=pn.cpu().tolist()
                        assert batch_preds==ref[split]['predictions'][count:count+len(batch_preds)],'Historical predictions changed'
                        preds.extend(batch_preds);correct+=int((pn==targets).sum());count+=len(targets)
                row=dict(condition=ref['name'],split=split,rows=rows,answers=count,accuracy=correct/count,
                         maximum_logit_absolute_error=max_error,logits_close=matching_logits,
                         logit_entries_over_tolerance=bad_entries,original_mean_nll=old_nll/count,new_mean_nll=new_nll/count,
                         predictions_equal_to_original_and_archived=True)
                save(out/(ref['name']+'-'+split+'-predictions.json'),preds)
                checks.append(row);total+=count
                print(json.dumps(row),flush=True)
                if a.logit_policy=='require':assert matching_logits,'Logit tolerance failed'
            del old_model,new_model,checkpoint
        assert all(sha(ROOT/p)==h for p,h in protected.items()),'Protected file changed'
        all_logits=all(r['logits_close'] for r in checks)
        result=dict(status='passed' if all_logits else 'predictions_match_but_logits_differ',
                    started_utc=started,finished_utc=now(),wall_seconds=time.perf_counter()-tick,
                    device=a.device,torch_version=torch.__version__,checks=checks,total_answers=total,
                    dtype=a.dtype,backend=a.backend,logit_policy=a.logit_policy,all_logits_close=all_logits,all_predictions_match=True,
                    optimizer_updates=0,new_generalization_evidence=False,query_chunk=a.query_chunk,
                    dropout='disabled during frozen checkpoint inference',protected_files_unchanged=True)
        save(out/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
    except Exception as e:
        save(out/'FAILURE.json',dict(utc=now(),error=str(e),checks=checks,answers_completed=total,optimizer_updates=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--batch',type=int,default=16)
    p.add_argument('--query-chunk',type=int,default=64);p.add_argument('--rows',type=int,default=0)
    p.add_argument('--max-seconds',type=float,default=600)
    p.add_argument('--dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--logit-policy',choices=['require','record'],default='require')
    p.add_argument('--backend',choices=['chunked','triton','triton_rank'],default='chunked')
    p.add_argument('--triton-gate',type=Path);p.add_argument('--selector-gate',type=Path)
    main(p.parse_args())
