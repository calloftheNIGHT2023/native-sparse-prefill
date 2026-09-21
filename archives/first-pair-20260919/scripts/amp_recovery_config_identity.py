"""Verify original config bytes and semantic equivalence of the saved snapshot."""
import hashlib,json
from pathlib import Path
def verify_config_identity(run_dir):
 run_dir=Path(run_dir).resolve()
 result=json.loads((run_dir/'result.json').read_text())
 source=run_dir.parents[2]/'data/flashmoba-amp-recovery-v0/config.json'
 snapshot=run_dir/'config.json'
 original_sha=hashlib.sha256(source.read_bytes()).hexdigest()
 assert original_sha==result['identity']['config_sha256'],'Original config bytes differ from training identity'
 assert json.loads(source.read_text())==json.loads(snapshot.read_text()),'Saved config values differ from original config'
 return dict(source_config=str(source),source_sha256=original_sha,snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),semantic_equal=True)
