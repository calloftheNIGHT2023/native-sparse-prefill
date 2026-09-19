"""Download a pinned public Qwen checkpoint; prepare independent within-model paired windows."""
import hashlib,json,shutil,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
out=ROOT/'data/flashmoba-qwen-precision-v0';out.mkdir(parents=True,exist_ok=False)
tick=time.perf_counter();started=utc()
with urllib.request.urlopen('https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B',timeout=30) as r:meta=json.load(r)
revision=meta['sha'];assert len(revision)==40
(out/'upstream-model-metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(dict(event='download_start',utc=utc(),revision=revision)),flush=True)
snapshot_download('Qwen/Qwen2.5-0.5B',revision=revision,local_dir=out/'model',token=False,
    allow_patterns=['*.json','*.safetensors','merges.txt','README.md'])
cfg=json.loads((ROOT/'configs/flashmoba-realtext-precision-v2.json').read_text())
cfg.update(model_id='Qwen/Qwen2.5-0.5B',model_revision=revision,model_dtype='bfloat16',
    numerical_revision_reason='Separate Qwen base-model numerical control after the Pythia70m baseline failed the unchanged .05 NLL adapter gate.',
    scope='Frozen Qwen2.5-0.5B base model, BF16 weights/QKV. All24 attention layers swapped for paired evaluation; not native sparse training, Qwen4, or a claimed original method. WikiText2 natural-newline windows;16 contexts with nested1024/2048 prefixes. No synthetic EOS inserted.')
corpus=Path('/workspace/flashmoba-wikitext-natural.txt')
cfg['raw_text_sha256']=sha(corpus)
tok=AutoTokenizer.from_pretrained(out/'model',local_files_only=True,trust_remote_code=False)
ids=tok(corpus.read_text(encoding='utf-8'),add_special_tokens=False)['input_ids'];width=max(cfg['lengths'])+1
windows=np.asarray(ids[:len(ids)//width*width],dtype=np.int64).reshape(-1,width)
selected=np.random.default_rng(cfg['selection_seed']).permutation(len(windows))[:cfg['contexts']]
assert len(selected)==cfg['contexts']
np.save(out/'tokens.npy',windows[selected]);(out/'config.json').write_text(json.dumps(cfg,indent=2))
shutil.copy2(__file__,out/'preparation-source.py')
manifest=dict(started_utc=started,finished_utc=utc(),seconds=time.perf_counter()-tick,model_id=cfg['model_id'],revision=revision,
    model_download_url=f'https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/{revision}',raw_text_sha256=sha(corpus),
    selected_nonoverlapping_windows=selected.tolist(),stream_token_count=len(ids),artificial_special_tokens=False,
    files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and '.cache' not in p.parts])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(event='ready',utc=utc(),revision=revision,contexts=len(selected),seconds=time.perf_counter()-tick)),flush=True)
