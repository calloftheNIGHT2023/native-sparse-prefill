"""Freeze paired 8K LoRA quality/cost controls before running any new outcomes."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/flashmoba-quality-cost-v0';OUT.mkdir(parents=True,exist_ok=False)
old=ROOT/'data/flashmoba-pool-training-v0'
cfg=json.loads((old/'config.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg['model_path'],local_files_only=True,trust_remote_code=False)
seed=2026091551;width=8193;arrays={};selected={}
for split in ['train','validation']:
    text=(old/f'{split}.txt').read_bytes()
    expected=next(x['text_sha256'] for x in cfg['sources'] if x['split']==split)
    assert hashlib.sha256(text).hexdigest()==expected
    ids=tokenizer(text.decode('utf-8'),add_special_tokens=False)['input_ids']
    windows=np.asarray(ids[:len(ids)//width*width],dtype=np.int64).reshape(-1,width)
    excluded=set()
    for prior,length in [('flashmoba-pool-training-v0',2048),('flashmoba-backward-training-v0',8192)]:
        meta=json.loads((ROOT/'results'/prior/'input-manifest.json').read_text())
        for index in meta['selected'][split]['indices']:
            lo=index*(length+1);hi=lo+length+1
            excluded.update(j for j in range(len(windows)) if j*width<hi and (j+1)*width>lo)
    available=np.asarray([j for j in range(len(windows)) if j not in excluded])
    order=np.random.default_rng(seed+(split=='validation')).permutation(available)
    if split=='train':
        assert len(order)>=64;arrays['train']=windows[order[:64]]
        selected['train']=dict(indices=order[:64].tolist(),excluded_prior_indices=sorted(excluded),corpus_tokens=len(ids))
    else:
        assert len(order)>=20
        for name,slice_ in [('calibration',slice(0,4)),('report',slice(4,20))]:
            arrays[name]=windows[order[slice_]]
            selected[name]=dict(indices=order[slice_].tolist(),excluded_prior_indices=sorted(excluded),corpus_tokens=len(ids))
np.savez(OUT/'tokens.npz',**arrays)
conditions=[dict(name='dense',extension='original',pool='dense',topk=0),
    dict(name='original_k4',extension='original',pool='official',topk=4),
    dict(name='barrier_k4',extension='barrier',pool='official',topk=4),
    dict(name='fp32_k4',extension='barrier',pool='fp32',topk=4),
    dict(name='fp32_k16',extension='barrier',pool='fp32',topk=16)]
cfg.update(seed=seed,initialization_seeds=[2026091552,2026091553],length=8192,steps=64,
    block_size=128,conditions=conditions,deterministic_backward=False,calibration_steps=[0,16,32,64],
    calibration_windows=4,report_windows=16,report_steps=[64],pool_config=[32,4,3],
    quality_screen_margin_nats=.03,training_time_ratio_screen=.95,
    scope='Two initialization seeds, identical token order across all conditions, 64 updates each. Paired exploratory LoRA quality/cost pilot; not native full-parameter pretraining. Report windows are held out from decisions within this pilot, not an untouched external dataset.',
    stop_rule='Stop on correctness/nonfinite/runtime failure; do not tune LR, steps or select K from report-window outcomes. No automatic scale-up unless prespecified quality/cost screen is met.',
    timing_scope='Synchronized wall time for data transfer, forward, backward, finite-gradient check and optimizer. Exclude preparation, model loading, checkpoint writes, calibration/report evaluation. Also retain complete wall time.',
    planned_trajectories=10,planned_optimizer_updates=640,train_tokens_per_trajectory=64*8192,
    trained_parameters='LoRA rank8 alpha16 in Q/K/V/O of all 24 layers; pretrained backbone frozen',
    prepared_utc=datetime.now(timezone.utc).isoformat(),selection=selected,
    tokens_sha256=hashlib.sha256((OUT/'tokens.npz').read_bytes()).hexdigest(),
    extension_sha256={'original':'b114a7755aad6556bc72eac1f8de0fcd0d5bd1e78d0dc1b1e521ac8621ce4d53','barrier':'72c3fda9e7d4da701bfc76b0e32f4d80fdbeec70d462c83b87b97f51d887ed1c'})
cfg.pop('comparison_pairs',None);cfg.pop('required_kernel_gates',None)
(OUT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(status='prepared',selection=selected,tokens_sha256=cfg['tokens_sha256'],planned_updates=640)),flush=True)
