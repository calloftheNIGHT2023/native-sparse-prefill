"""Verify the cloud archive, reject path escapes and conflicts, then restore evidence."""
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'exports/flashmoba-mixed-cost-cloud-stage-v0.tar.gz'
proof = json.loads((ROOT / 'exports/flashmoba-mixed-cost-cloud-stage-v0.json').read_text(encoding='utf-8'))
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest == proof['sha256'] and archive.stat().st_size == proof['bytes']
with tarfile.open(archive, 'r:gz') as tar:
    members = tar.getmembers()
    assert len(members) == proof['members']
    assert len({m.name for m in members}) == len(members)
    by_name = {m.name: m for m in members}
    for member in members:
        assert member.isfile() and '\\' not in member.name
        destination = (ROOT / member.name).resolve()
        assert destination.is_relative_to(ROOT.resolve())
    manifest_name = 'results/flashmoba-mixed-cost-package-v0/archive-manifest.json'
    manifest = json.load(tar.extractfile(by_name[manifest_name]))
    assert set(by_name) == {r['path'] for r in manifest} | {manifest_name}
    for row in manifest:
        raw = tar.extractfile(by_name[row['path']]).read()
        assert len(raw) == row['bytes']
        assert hashlib.sha256(raw).hexdigest() == row['sha256'], row['path']
    # Restore only absent or byte-identical files; never silently overwrite local edits.
    for member in members:
        destination = ROOT / member.name
        if destination.exists():
            assert destination.read_bytes() == tar.extractfile(member).read(), f'Local conflict: {member.name}'
    tar.extractall(ROOT, filter='data')
for row in manifest:
    path = ROOT / row['path']
    assert path.stat().st_size == row['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
result = dict(status='verified',verified_utc=datetime.now(timezone.utc).isoformat(),
              archive=str(archive),sha256=digest,bytes=archive.stat().st_size,
              members=len(members),manifest_files=len(manifest),
              validation='Archive digest, all member digests before extraction, path safety, local conflict check, all restored file digests.')
out = ROOT / 'provenance/flashmoba-mixed-cost-local-verification-v0.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
