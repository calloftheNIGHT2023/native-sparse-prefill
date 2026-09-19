import hashlib, json, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'provenance/cloud-connect-2026-09-14'
out.mkdir(exist_ok=True)
files=[]
for directory in ['src','tests','configs','scripts']:
    files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files.extend(p for p in (ROOT/'data/realtext-early-step1000/assets/model').iterdir() if p.is_file())
files.append(ROOT/'data/realtext-early-step1000/download-manifest.json')
manifest=[dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(files)]
(out/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(out/'bundle.tar','w') as tar:
    for p in files: tar.add(p,arcname=p.relative_to(ROOT).as_posix())
    tar.add(out/'bundle-manifest.json',arcname='logs/upload-manifest.json')
print(json.dumps(dict(files=len(files),bytes=(out/'bundle.tar').stat().st_size)))
