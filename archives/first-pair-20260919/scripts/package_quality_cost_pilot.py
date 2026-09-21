"""Preserve all completed and failed attempts, step logs, checkpoints and inputs."""
import hashlib,json,tarfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-quality-cost-package-v0';OUT.mkdir(parents=True,exist_ok=False)
assert json.loads((ROOT/'results/flashmoba-quality-cost-analysis-v0/result.json').read_text())['status']=='complete'
(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
files=set()
for p in (ROOT/'results').glob('flashmoba-quality-cost-*'):
    if p.is_dir():files.update(f for f in p.rglob('*') if f.is_file())
files.update((ROOT/'scripts').glob('*quality_cost*.py'))
files.update(f for f in (ROOT/'data/flashmoba-quality-cost-v0').rglob('*') if f.is_file())
rows=[dict(path=f.relative_to(ROOT).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(files)]
mf=OUT/'archive-manifest.json';mf.write_text(json.dumps(rows,indent=2)+'\n')
p=Path('/workspace/flashmoba-quality-cost-cloud-stage-v0.tar.gz');assert not p.exists()
with tarfile.open(p,'w:gz') as t:
    for f in sorted(files | {mf}):t.add(f,arcname=f.relative_to(ROOT).as_posix())
proof=dict(created_utc=datetime.now(timezone.utc).isoformat(),archive=str(p),bytes=p.stat().st_size,
           sha256=hashlib.sha256(p.read_bytes()).hexdigest(),members=len(files)+1)
Path('/workspace/flashmoba-quality-cost-cloud-stage-v0.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof),flush=True)
