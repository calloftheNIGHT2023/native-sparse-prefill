"""Verify cloud evidence and wheel bytes before restoring locally; never overwrite different files."""
import hashlib,json,tarfile,zipfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
archive=ROOT/'exports/flashmoba-cloud-results-v0.tar.gz'
proof=json.loads((ROOT/'provenance/flashmoba-cloud-export-proof-v0.json').read_text())
assert sha(archive)==proof['sha256']
with tarfile.open(archive) as tar:
    members=tar.getmembers();assert len(members)==proof['files'] and len({m.name for m in members})==len(members)
    manifest_name='provenance/flashmoba-cloud-results-v0.json';manifest=json.load(tar.extractfile(manifest_name))
    records={r['path']:r for r in manifest['files']};assert len(records)+1==len(members)
    for m in members:
        p=PurePosixPath(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts
        target=(ROOT/m.name).resolve();assert ROOT in target.parents
        blob=tar.extractfile(m).read();digest=hashlib.sha256(blob).hexdigest()
        if m.name!=manifest_name:assert digest==records[m.name]['sha256'] and len(blob)==records[m.name]['bytes']
        if target.exists():assert sha(target)==digest,('Existing file differs; no overwrite',m.name)
    for m in members:
        target=ROOT/m.name
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(m).read())
for p,rec in records.items():assert sha(ROOT/p)==rec['sha256']
wheel_manifest=json.loads((ROOT/'exports/flashmoba-wheels-v0/manifest.json').read_text())
wheel=ROOT/wheel_manifest['files'][0]['path'];assert sha(wheel)==wheel_manifest['files'][0]['sha256']
with zipfile.ZipFile(wheel) as z:
    extension=[n for n in z.namelist() if n.startswith('flash_moba_cuda') and n.endswith('.so')];assert len(extension)==1
    binary_hash=hashlib.sha256(z.read(extension[0])).hexdigest()
gate=json.loads((ROOT/'results/flashmoba-official-gate-v1/manifest.json').read_text())
assert binary_hash==gate['extension_sha256']
result=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),archive_sha256=sha(archive),files=len(members),
    all_members_and_local_files_verified=True,wheel_sha256=sha(wheel),extension_sha256=binary_hash,
    wheel_binary_equals_gpu_tested_extension=True,new_machine_install_not_tested=True)
(ROOT/'provenance/flashmoba-local-evidence-proof-v0.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
