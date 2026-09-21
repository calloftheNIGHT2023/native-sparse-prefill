"""Archive successful or failed bounded stages, including every saved checkpoint."""
import json,hashlib,tarfile,io
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def package(out):
 out=Path(out).resolve();assert R.resolve() in out.parents
 archive=R/'exports/32k-adaptation-evidence-v0.tar.gz';assert not archive.exists()
 files=[p for p in out.rglob('*') if p.is_file()]
 protocol=json.loads((R/'provenance/32k-adaptation-protocol.json').read_text())
 files += [R/n for n in protocol['source_sha256']]+list((R/'data/32k-adaptation-v0').glob('*'))+[R/'provenance/32k-adaptation-protocol.json',R/'docs/32k-adaptation-protocol-2026-09-16.md']
 entries=[]
 with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
  for p in sorted(set(files)):
   if not p.is_file():continue
   raw=p.read_bytes();name=p.relative_to(R).as_posix();entries.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
   info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries),indent=2).encode();info=tarfile.TarInfo('32k-adaptation-evidence-manifest.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 proof=dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(entries))
 archive.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
