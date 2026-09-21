"""Preserve the public primary sources used in this stage's overlap screen."""
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'literature/block-candidate-2026-09-15';out.mkdir(parents=True,exist_ok=False)
sources=[('flashmoba.html','https://arxiv.org/html/2511.11571v1'),
         ('moba.html','https://arxiv.org/html/2502.13189v1'),
         ('quest.html','https://arxiv.org/html/2406.10774v1'),
         ('uncertainty-gated.html','https://arxiv.org/html/2607.07724v1')]
rows=[]
for name,url in sources:
    req=urllib.request.Request(url,headers={'User-Agent':'NativeSparseResearch/0.1'})
    with urllib.request.urlopen(req,timeout=45) as response:blob=response.read()
    (out/name).write_bytes(blob)
    rows.append(dict(path=name,url=url,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),
        downloaded_utc=datetime.now(timezone.utc).isoformat()))
    (out/'sources.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows[-1]),flush=True)
