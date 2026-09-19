"""Package the active router study and preserve an explicit inclusion/hash manifest."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
paths=set()
for name in ['src','scripts','tests','third_party','configs','docs','literature']:
    paths.update(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for name in ['schedule-screen-cloud-v0','schedule-screen-analysis-v0','schedule-checkpoint-audit-v0','frozen-router-capacity-v0','frozen-router-audit-v0','joint-token-router-v0','joint-token-analysis-v0']:
    paths.update(p for p in (ROOT/'results'/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for pattern in ['frozen*','joint-token*','migration*','sparse-schedule*','router-literature*','control-state.json']:
    paths.update(p for p in (ROOT/'logs').glob(pattern) if p.is_file())
paths.update(p for p in (ROOT/'logs/schedule-screen-cloud-v0').rglob('*') if p.is_file())
for name in ['STATE.md','TIMELINE.md','README.md','requirements-schedule-cloud.txt','requirements-router-portable.txt']:paths.add(ROOT/name)
for p in paths:
    assert not p.is_symlink() and not p.name.startswith('.env') and p.suffix not in ('.pem','.key')
manifest=dict(scope='All current source plus active schedule/frozen/joint router datasets, checkpoints and evidence. Historical 70M artifacts remain on the local computer and are not needed for this study. No virtualenv or private key.',files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)])
mf=ROOT/'exports/router-migration-manifest-v1.json';mf.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
out=ROOT/'exports/router-migration-v1.tar.gz'
with tarfile.open(out,'w:gz',compresslevel=1) as tar:
    for p in sorted(paths):tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    tar.add(mf,arcname='MIGRATION_MANIFEST.json')
with tarfile.open(out,'r:gz') as tar:
    for row in manifest['files']:
        f=tar.extractfile(row['path']);h=hashlib.sha256()
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
        assert h.hexdigest()==row['sha256'],row['path']
record=dict(archive=out.name,sha256=sha(out),compressed_bytes=out.stat().st_size,uncompressed_bytes=sum(r['bytes'] for r in manifest['files']),files=len(paths),archive_members_hash_verified=True,scope=manifest['scope'])
(ROOT/'exports/router-migration-v1.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record))
