"""Snapshot every successful or failed run and its reproducible inputs."""
import hashlib,json,tarfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-mixed-cost-package-v0';OUT.mkdir(parents=True,exist_ok=False)
assert (ROOT/'results/flashmoba-mixed-cost-analysis-v0/result.json').exists()
(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
files=set()
for folder in (ROOT/'results').glob('flashmoba-mixed-cost-*'):
    if folder.is_file():files.add(folder)
    else:files.update(p for p in folder.rglob('*') if p.is_file())
files.update((ROOT/'scripts').glob('*mixed*cost*.py'))
files.update((ROOT/'configs').glob('flashmoba-mixed-cost-v*.json'))
files.update((ROOT/'provenance').glob('flashmoba-mixed-cost-*.json'))
for name in ['configs/flashmoba-mixed-cost-v0.json','configs/flashmoba-mixed-cost-v1.json',
             'docs/flashmoba-mixed-cost-protocol-2026-09-15.md','docs/flashmoba-mixed-cost-v1-protocol-2026-09-15.md',
             'scripts/run_pool_training_control.py','scripts/run_flashmoba_realtext_precision.py',
             'scripts/verify_flashmoba_official_v1.py','data/flashmoba-quality-cost-v0/tokens.npz',
             'data/flashmoba-quality-cost-v0/config.json','provenance/flashmoba-mixed-cost-input-v0.json']:
    files.add(ROOT/name)
rows=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
mf=OUT/'archive-manifest.json';mf.write_text(json.dumps(rows,indent=2)+'\n')
archive=Path('/workspace/flashmoba-mixed-cost-cloud-stage-v0.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(files|{mf}):t.add(p,arcname=p.relative_to(ROOT).as_posix())
proof=dict(utc=datetime.now(timezone.utc).isoformat(),archive=str(archive),bytes=archive.stat().st_size,
           sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=len(files)+1,manifest_files=len(rows))
Path('/workspace/flashmoba-mixed-cost-cloud-stage-v0.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
