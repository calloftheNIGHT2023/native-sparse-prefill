"""Resume final integrity check after supplying two missing source archives; no scientific changes."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected='254dbba89d67cbf785aa1f151d8d0ce9a43e5dc28a589f32aa882c30c8ed25f8';assert sha(root/'exports/router-author-session-final-v0.tar.gz')==expected;dest=root/'provenance/router-author-session-final-2026-09-14';assert dest.is_dir();manifest=json.loads((dest/'manifest.json').read_text(encoding='utf-8'));raw=json.loads((dest/'raw-bundles.json').read_text(encoding='utf-8'))
for row in manifest:assert sha(dest/row['path'])==row['sha256']
for row in raw:assert sha(root/row['path'])==row['sha256']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip();copied=0
for row in manifest:
    if row['path']=='raw-bundles.json':continue
    source=dest/row['path'];target=(root/row['path']).resolve();assert target.is_relative_to(root.resolve());target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert sha(target)==row['sha256'];copied+=1
result=dict(utc=datetime.now(timezone.utc).isoformat(),archive_sha256=expected,manifest_files_verified=len(manifest),raw_archives_verified=len(raw),live_files_copied_and_verified=copied,gpu_compute_empty=True,pod_stopped=False,backup_recovery_note='First check found the two author-control source archives missing under cloud exports. Both were uploaded from their hash-verified local originals; existing extraction and all nine raw archives now pass.',verification_script_sha256=sha(Path(__file__)))
out=root/'logs/router-author-final-cloud-verification.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
