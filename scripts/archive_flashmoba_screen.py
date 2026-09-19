"""Archive primary sources for the official-baseline and boundary-overlap screen."""
import hashlib,json,urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'literature/flashmoba-baseline-2026-09-15';out.mkdir(parents=True,exist_ok=False)
sources=[('longlora.html','https://arxiv.org/html/2309.12307v2'),
    ('pbs-attn.html','https://arxiv.org/html/2510.21270v1'),
    ('boundary-repair.html','https://arxiv.org/html/2606.02680v1'),
    ('cuda-minor-compatibility.html','https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html')]
rows=[]
for name,url in sources:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'NativeSparseResearch/0.1'}),timeout=45) as r:blob=r.read()
    (out/name).write_bytes(blob)
    rows.append(dict(path=name,url=url,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),utc=datetime.now(timezone.utc).isoformat()))
    (out/'sources.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows[-1]),flush=True)
