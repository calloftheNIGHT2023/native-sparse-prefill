"""Package completed task stage, locked inputs and code, with byte hashes."""
import hashlib,io,json,tarfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
files=set()
for sub in ['results/task-quality-stage-v0','results/task-runtime-stage-v0','results/task-layers-stage-v0','results/task-factorial-stage-v0','data/task-quality-v0']:
 files.update(p for p in (R/sub).rglob('*') if p.is_file())
for pattern in ['scripts/*task*.py','scripts/amp_recovery_config_identity.py','scripts/run_amp_recovery.py','scripts/amp_recovery_state.py','scripts/chunked_lm_loss.py','scripts/run_flashmoba_realtext_precision.py','docs/task-quality-protocol-2026-09-15.md','docs/task-runtime-protocol-2026-09-15.md','docs/task-layers-protocol-2026-09-15.md','docs/task-factorial-protocol-2026-09-15.md','provenance/task-quality*']:
 files.update(p for p in R.glob(pattern) if p.is_file())
a=R/'exports/task-factorial-evidence-v0.tar.gz';assert not a.exists();rows=[]
with tarfile.open(a,'w:gz') as t:
 for p in sorted(files):
  raw=p.read_bytes();name=p.relative_to(R).as_posix();rows.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));m=tarfile.TarInfo(name);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 raw=json.dumps(dict(files=rows,completed_job_directories=[]),indent=2).encode();m=tarfile.TarInfo('amp-recovery-results-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
proof=dict(sha256=hashlib.sha256(a.read_bytes()).hexdigest(),bytes=a.stat().st_size,files=len(rows));a.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
