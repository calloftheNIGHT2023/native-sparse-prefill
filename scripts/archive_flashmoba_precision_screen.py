"""Primary sources for a scoped precision/uncertainty overlap check, not full review."""
import hashlib,json,urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'literature/flashmoba-precision-screen-2026-09-15';out.mkdir(parents=True,exist_ok=False)
rows=[]
for name,url in [('switch-v3.html','https://arxiv.org/html/2101.03961v3'),
                 ('uncertainty-router.html','https://github.com/ThomasRossi/uncertainty-gated-block-sparse-attention'),
                 ('spotattention-abs.html','https://arxiv.org/abs/2606.22874'),
                 ('stream-id-check.html','https://arxiv.org/abs/2510.19875')]:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'NativeSparseResearch/0.1'}),timeout=40) as r:blob=r.read()
    (out/name).write_bytes(blob)
    rows.append(dict(path=name,url=url,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),utc=datetime.now(timezone.utc).isoformat()))
    (out/'sources.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]),flush=True)
