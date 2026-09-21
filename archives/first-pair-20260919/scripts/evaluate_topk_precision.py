"""Frozen-model precision audit on ALREADY EXPOSED MQAR evaluation data."""
import argparse,hashlib,json,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from router_author_control import make_model,selected_logits
from zoology_sparse_schedule import install


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    from triton_topk_selector import TritonRankTopKAttention
    for file,source,count in [(a.aggregation_gate,'src/triton_selected_attention.py',30),
                              (a.selector_gate,'src/triton_topk_selector.py',15)]:
        gate=json.loads(file.read_text());assert gate['status']=='passed' and len(gate['checks'])==count
        assert gate['sources'][source]==sha(ROOT/source)
    folder=ROOT/'results/router-author-falsification-evaluation-v0'
    data=torch.load(folder/'evaluation-data.pt',map_location='cpu',weights_only=True)
    refs=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))['conditions']
    checkpoints=['results/router-author-falsification-lowlr-v0/checkpoint.pt',
                 'results/router-author-falsification-confirm-v0/checkpoint.pt']
    protected={p:sha(ROOT/p) for p in checkpoints+['scripts/evaluate_topk_precision.py',
        'src/triton_selected_attention.py','src/triton_topk_selector.py','src/zoology_sparse_schedule.py',
        (folder/'evaluation.json').relative_to(ROOT).as_posix(),(folder/'evaluation-data.pt').relative_to(ROOT).as_posix()]}
    for path in protected:
        if path.endswith('.py'):
            dst=out/'source'/path;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/path).read_bytes())
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(4)
    started=now();tick=time.perf_counter();checks=[]
    try:
        for path in checkpoints:
            reference=next(r for r in refs if r['checkpoint']==path)
            assert sha(ROOT/path)==reference['checkpoint_sha256']
            ckpt=torch.load(ROOT/path,map_location='cpu',weights_only=False)
            for dtype in ['float32','bfloat16']:
                for method in ['legacy_topk','triton_rank_topk']:
                    m=make_model(ckpt['config'],ckpt['model'],'cuda').to(getattr(torch,dtype))
                    install(m,'native')
                    if method=='triton_rank_topk':
                        for layer in m.backbone.layers:
                            layer.sequence_mixer.inner_attn=TritonRankTopKAttention(query_chunk=256)
                    m.eval()
                    for split in ['test','swapped','noisy']:
                        part=data[split];pred=[];correct=0;count=0;nll=0.
                        with torch.no_grad():
                            for first in range(0,len(part['inputs']),a.batch):
                                logits,targets=selected_logits(m,part['inputs'][first:first+a.batch].cuda(),
                                    part['labels'][first:first+a.batch].cuda())
                                assert bool(torch.isfinite(logits).all())
                                p=logits.argmax(-1);pred.extend(p.cpu().tolist());correct+=int((p==targets).sum());count+=len(targets)
                                nll+=float(torch.nn.functional.cross_entropy(logits.float(),targets,reduction='sum'))
                        old=reference[split]['predictions'];assert len(old)==count
                        row=dict(checkpoint=path,dtype=dtype,method=method,split=split,answers=count,
                            accuracy=correct/count,mean_nll=nll/count,
                            prediction_disagreements_from_archived_fp32=sum(x!=y for x,y in zip(pred,old)),
                            accuracy_delta_from_archived_fp32=correct/count-reference[split]['accuracy'])
                        checks.append(row);print(json.dumps(row),flush=True)
                        (out/(Path(path).parent.name+'-'+dtype+'-'+method+'-'+split+'-predictions.json')).write_text(json.dumps(pred))
                    del m
        status='complete';error=None
    except Exception as exc:
        status='incomplete';error=dict(message=str(exc),traceback=traceback.format_exc())
    result=dict(status=status,error=error,started_utc=started,finished_utc=now(),wall_seconds=time.perf_counter()-tick,
        checks=checks,sources=protected,protected_files_unchanged=all(sha(ROOT/p)==h for p,h in protected.items()),
        scientific_optimizer_updates=0,new_generalization_evidence=False,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        limitations=['Frozen inference only; not BF16 training stability','Already exposed toy MQAR data',
                     'Accuracy changes are measured, not converted into a pass by changing a tolerance'])
    (out/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if status!='complete' or not result['protected_files_unchanged']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--aggregation-gate',type=Path,required=True);p.add_argument('--selector-gate',type=Path,required=True)
    main(p.parse_args())
