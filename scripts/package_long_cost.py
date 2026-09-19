"""Snapshot every successful or failed run and its reproducible inputs."""
import hashlib,json,tarfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-long-cost-package-v0';OUT.mkdir(parents=True,exist_ok=False)
assert (ROOT/'results/flashmoba-long-cost-controller-v0/result.json').exists()
(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
files=set()
for folder in (ROOT/'results').glob('flashmoba-long-cost-*'):
    if folder.is_file():files.add(folder)
    else:files.update(p for p in folder.rglob('*') if p.is_file())
for name in ['scripts/run_long_context_cost.py','scripts/run_long_cost_stage.py','scripts/chunked_lm_loss.py','scripts/check_chunked_loss_cpu.py','configs/flashmoba-long-cost-v0.json','docs/flashmoba-long-cost-protocol-2026-09-15.md','provenance/flashmoba-long-cost-input-v0.json','data/flashmoba-quality-cost-v0/tokens.npz','data/flashmoba-quality-cost-v0/config.json','data/flashmoba-qwen-precision-v0/manifest.json']:
    files.add(ROOT/name)
files.update((ROOT/'provenance').glob('flashmoba-long-cost-recovery*'))
rows=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
mf=OUT/'archive-manifest.json';mf.write_text(json.dumps(rows,indent=2)+'\n')
archive=Path('/workspace/flashmoba-long-cost-cloud-stage-v0.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(files|{mf}):t.add(p,arcname=p.relative_to(ROOT).as_posix())
proof=dict(utc=datetime.now(timezone.utc).isoformat(),archive=str(archive),bytes=archive.stat().st_size,
           sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=len(files)+1,manifest_files=len(rows))
Path('/workspace/flashmoba-long-cost-cloud-stage-v0.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
