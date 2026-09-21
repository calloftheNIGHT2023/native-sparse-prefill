"""Additional primary-source overlap check; separate artifact from earlier completed archive."""
import hashlib,json,urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'literature/flashmoba-rope-screen-2026-09-15';out.mkdir(parents=True,exist_ok=False)
rows=[]
for name,url in [('prism-v2.html','https://arxiv.org/html/2602.08426v2'),('hga-v1.html','https://arxiv.org/html/2606.30709v1')]:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'NativeSparseResearch/0.1'}),timeout=45) as r:blob=r.read()
    (out/name).write_bytes(blob)
    rows.append(dict(path=name,url=url,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),utc=datetime.now(timezone.utc).isoformat()))
    (out/'sources.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]),flush=True)
