"""Package the final report, plots, state and provenance without altering raw results."""
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
names = ['STATE.md', 'TIMELINE.md', 'logs/control-state.json', 'docs/flashmoba-long-cost-protocol-2026-09-15.md', 'docs/flashmoba-long-cost-results-2026-09-15.md', 'docs/flashmoba-long-cost-run-index-2026-09-15.md', 'docs/flashmoba-long-cost-stop-and-resume-2026-09-15.md', 'provenance/flashmoba-long-cost-local-verification-v0.json', 'provenance/flashmoba-long-cost-local-recovery-verification-v0.json', 'exports/flashmoba-long-cost-cloud-stage-v0.json', 'scripts/analyze_long_cost.py', 'scripts/verify_long_cost_backup.py', 'results/flashmoba-long-cost-analysis-v0/result.json', 'results/flashmoba-long-cost-analysis-v0/source.py']
names = sorted(set(names))
manifest = [dict(path=n,bytes=(ROOT/n).stat().st_size,
                 sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest()) for n in names]
m = ROOT/'provenance/flashmoba-long-cost-final-overlay-manifest-v0.json'
assert not m.exists()
m.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
names.append(m.relative_to(ROOT).as_posix())
archive = ROOT/'exports/flashmoba-long-cost-final-overlay-v0.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for n in names:tar.add(ROOT/n,arcname=n)
proof = dict(created_utc=datetime.now(timezone.utc).isoformat(),archive=archive.name,
             bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
             members=len(names),manifest_files=len(manifest),
             cloud_verification_record='provenance/flashmoba-long-cost-final-cloud-verification-v0.json')
(ROOT/'exports/flashmoba-long-cost-final-overlay-v0.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
print(json.dumps(proof,indent=2))
