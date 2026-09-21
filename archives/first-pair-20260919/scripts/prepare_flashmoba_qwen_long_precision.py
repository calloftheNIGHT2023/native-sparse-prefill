"""Finite8K follow-up, using the same pinned model and corpus; no training."""
import json,hashlib,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
out=ROOT/'data/flashmoba-qwen-long-precision-v0';out.mkdir(parents=True,exist_ok=False)
start=utc();tick=time.perf_counter()
model=ROOT/'data/flashmoba-qwen-precision-v0/model'
(out/'model').symlink_to(model,target_is_directory=True)
c=json.loads((ROOT/'configs/flashmoba-qwen-precision-v1.json').read_text())
c.update(lengths=[8192],selection_seed=2026091540,probe_contexts=2,
         model_source='data/flashmoba-qwen-precision-v0/model',dataset_source='/workspace/flashmoba-wikitext-natural.txt',
         decision='Exploratory8K boundary check after2K negative result. Same absolute .01 NLL and context-bootstrap95CI threshold, both K values required. Not an independent confirmatory study or a training authorization trigger by itself.',
         scope='Frozen Qwen2.5-0.5B FP32 backbone/BF16 attention,16 nonoverlapping8193-token windows from the same WikiText2 test corpus. Prior shorter windows can overlap these long windows. No optimizer updates, no original algorithm claim, not Qwen4.')
tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
raw=Path('/workspace/flashmoba-wikitext-natural.txt');assert sha(raw)==c['raw_text_sha256']
ids=tok(raw.read_text(encoding='utf-8'),add_special_tokens=False)['input_ids'];width=8193
windows=np.asarray(ids[:len(ids)//width*width],dtype=np.int64).reshape(-1,width)
chosen=np.random.default_rng(c['selection_seed']).permutation(len(windows))[:c['contexts']]
assert len(chosen)==16
np.save(out/'tokens.npy',windows[chosen]);(out/'config.json').write_text(json.dumps(c,indent=2));shutil.copy2(__file__,out/'preparation-source.py')
manifest=dict(started_utc=start,finished_utc=utc(),seconds=time.perf_counter()-tick,shared_model='data/flashmoba-qwen-precision-v0/model',
    shared_model_manifest_sha256=sha(ROOT/'data/flashmoba-qwen-precision-v0/manifest.json'),chosen_windows=chosen.tolist(),
    files=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(utc=utc(),status='ready',contexts=len(chosen),length=8192)))
