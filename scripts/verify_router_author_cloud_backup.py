"""Run on the existing cloud workspace after final archive upload; print only integrity status."""
import argparse,hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--archive-sha',required=True);args=p.parse_args();assert len(args.archive_sha)==64 and all(c in '0123456789abcdef' for c in args.archive_sha)
root=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
bundle=root/'exports/router-author-session-final-v0.tar.gz';assert sha(bundle)==args.archive_sha;dest=root/'provenance/router-author-session-final-2026-09-14';assert not dest.exists()
with tarfile.open(bundle) as t:t.extractall(root,filter='data')
manifest=json.loads((dest/'manifest.json').read_text(encoding='utf-8'));raw=json.loads((dest/'raw-bundles.json').read_text(encoding='utf-8'))
for row in manifest:assert sha(dest/row['path'])==row['sha256'],row['path']
for row in raw:assert sha(root/row['path'])==row['sha256'],row['path']
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip();assert not gpu,'GPU compute still active'
copied=0
for row in manifest:
    if row['path']=='raw-bundles.json':continue
    rel=Path(row['path']);source=dest/rel;target=(root/rel).resolve();assert target.is_relative_to(root.resolve());target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert sha(target)==row['sha256'];copied+=1
result=dict(utc=datetime.now(timezone.utc).isoformat(),archive_sha256=args.archive_sha,manifest_files_verified=len(manifest),raw_archives_verified=len(raw),live_files_copied_and_verified=copied,gpu_compute_empty=True,pod_stopped=False)
target=root/'logs/router-author-final-cloud-verification.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
