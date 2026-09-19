"""Read a frozen plan, run every listed condition, preserve all predictions."""
import argparse
import gzip
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from router_author_control import make_model, selected_logits
from block_candidate_topk import BlockCandidateTopKAttention
from triton_topk_selector import TritonRankTopKAttention


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x): p.write_text(json.dumps(x, indent=2), encoding='utf-8')


def main(a):
    out = a.output; out.mkdir(parents=True, exist_ok=False)
    start = utc(); tick = time.perf_counter(); rows = []
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    plan = json.loads((a.data / 'preregistration.json').read_text())
    audit = json.loads((a.data / 'data-audit.json').read_text()); assert sha(a.data / 'evaluation-data.pt') == audit['data_sha256']
    sources = ['src/block_candidate_topk.py', 'src/triton_topk_selector.py', 'src/triton_selected_attention.py',
               'scripts/confirm_block_candidate_topk.py']
    protected = {p: sha(ROOT / p) for p in sources + plan['checkpoint_paths']}
    if a.fused_gate:
        from triton_block_candidate_topk import FusedBlockCandidateTopKAttention
        gate=json.loads(a.fused_gate.read_text())
        assert gate['status']=='passed' and len(gate['checks'])==30
        assert gate['sources']['src/triton_block_candidate_topk.py']==sha(ROOT/'src/triton_block_candidate_topk.py')
        sources.append('src/triton_block_candidate_topk.py');protected[sources[-1]]=sha(ROOT/sources[-1])
    for path in sources:
        dst=out/'source'/path; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes((ROOT/path).read_bytes())
    save(out / 'manifest.json', dict(started_utc=start, sources=protected, plan=plan, data_audit=audit,
        plan_sha256=sha(a.data / 'preregistration.json'), torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        dtype=a.dtype,implementation='fused implementation replay on now-exposed confirmation data' if a.fused_gate else 'PyTorch candidate confirmation'))
    data = torch.load(a.data / 'evaluation-data.pt', map_location='cpu', weights_only=True)
    def event(x):
        with (out / 'events.jsonl').open('a') as f: f.write(json.dumps(dict(utc=utc(), elapsed_seconds=time.perf_counter()-tick, **x))+'\n')
        print(json.dumps(x),flush=True)
    try:
        for path in plan['checkpoint_paths']:
            assert sha(ROOT / path) == plan['checkpoints'][path]
            ckpt=torch.load(ROOT/path, map_location='cpu', weights_only=False)
            model=make_model(ckpt['config'],ckpt['model'],'cuda').to(getattr(torch,a.dtype)).eval(); refs={}
            for cfg in plan['configs']:
                begin=utc(); ct=time.perf_counter()
                for i,layer in enumerate(model.backbone.layers):
                    layer.sequence_mixer.inner_attn=(TritonRankTopKAttention(query_chunk=256) if cfg['method']=='exact' or i not in cfg['layers'] else
                        BlockCandidateTopKAttention(**{k:cfg[k] for k in ['method','block','routes','offset']},backend='triton',query_chunk=32)).eval()
                    if a.fused_gate and cfg['method']!='exact' and i in cfg['layers']:
                        layer.sequence_mixer.inner_attn=FusedBlockCandidateTopKAttention(**{k:cfg[k] for k in ['method','block','routes','offset']}).eval()
                for split in ['test','swapped','noisy']:
                    part=data[split]; pred=[]; corrects=[]; ns=[]; nll=0.
                    with torch.no_grad():
                        for first in range(0,len(part['inputs']),32):
                            logits, labels=selected_logits(model,part['inputs'][first:first+32].cuda(),part['labels'][first:first+32].cuda())
                            assert bool(logits.isfinite().all()); p=logits.argmax(-1); pred.extend(p.cpu().tolist())
                            counts=(part['labels'][first:first+32]!=-100).sum(-1).tolist(); cursor=0
                            for number in counts:
                                corrects.append(int((p[cursor:cursor+number]==labels[cursor:cursor+number]).sum())); ns.append(number); cursor+=number
                            nll+=float(torch.nn.functional.cross_entropy(logits.float(),labels,reduction='sum'))
                    if cfg['method']=='exact': refs[split]=pred
                    row=dict(checkpoint=path,config=cfg,split=split,accuracy=sum(corrects)/sum(ns),answers=sum(ns),nll=nll/sum(ns),
                        prediction_disagreements=sum(x!=y for x,y in zip(pred,refs[split])),correct_by_sequence=corrects,answers_by_sequence=ns)
                    rows.append(row)
                    with gzip.open(out/(Path(path).parent.name+'-'+cfg['name']+'-'+split+'.json.gz'),'wt') as f: json.dump(pred,f)
                    event({k:v for k,v in row.items() if k not in ['correct_by_sequence','answers_by_sequence']})
                event(dict(event='condition_complete',checkpoint=path,config=cfg,started_utc=begin,finished_utc=utc(),wall_seconds=time.perf_counter()-ct))
                save(out/'partial.json',rows)
            del model
        status='complete'; error=None
    except Exception as exc:
        status='incomplete'; error=dict(message=str(exc),traceback=traceback.format_exc()); event(dict(event='failed',error=error))
    save(out/'evaluation.json',dict(status=status,error=error,started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,
        conditions=rows,dtype=a.dtype,scientific_optimizer_updates=0,protected_files_unchanged=all(sha(ROOT/p)==h for p,h in protected.items()),
        limitations=['N256 synthetic MQAR only', 'No text training, no speed conclusion', 'Configurations selected on earlier exposed data before this fresh evaluation']))
    if status!='complete': raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fused-gate',type=Path)
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32');main(p.parse_args())
