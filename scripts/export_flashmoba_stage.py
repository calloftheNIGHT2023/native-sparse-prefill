"""Explicit incremental evidence + compiled wheel export; never include envs or keys."""
import hashlib,json,tarfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
archive=Path('/workspace/flashmoba-cloud-results-v0.tar.gz');assert not archive.exists()
files=[]
for pattern in ['results/flashmoba-*','exports/flashmoba-wheels-v0']:
    for folder in ROOT.glob(pattern):files.extend(p for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
for pattern in ['logs/flashmoba-*.log','scripts/*flashmoba*.py','src/flashmoba_fixed_metadata.py']:
    files.extend(p for p in ROOT.glob(pattern) if p.is_file() and not p.is_symlink())
files=sorted(set(files))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files]
rel='provenance/flashmoba-cloud-results-v0.json';manifest=ROOT/rel;assert not manifest.exists()
manifest.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),type='additive_evidence_and_wheel_overlay',files=records,
    requires_previous_project_modules=True,contains_training_weights=False,contains_driver=False,contains_credentials=False),indent=2))
with tarfile.open(archive,'w:gz',compresslevel=2) as tar:
    for p in files+[manifest]:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
with tarfile.open(archive) as tar:
    assert len(tar.getmembers())==len(records)+1
    for rec in records:assert hashlib.sha256(tar.extractfile(rec['path']).read()).hexdigest()==rec['sha256']
proof=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(records)+1,all_members_verified=True)
(ROOT/'provenance/flashmoba-cloud-export-proof-v0.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
