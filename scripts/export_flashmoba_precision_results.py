"""Export completed precision work with explicit public-model exclusions and hashes."""
import argparse,hashlib,json,tarfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--version',choices=['v0','v1'],default='v0');version=p.parse_args().version
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
selected=set()
for pat in ['results/flashmoba-realtext-precision-*/**/*','results/flashmoba-qwen*-precision-*/**/*',
            'results/flashmoba-qwen-long-replay-v0/**/*',
            'results/flashmoba-fixed-pool-*/**/*','results/flashmoba-pool-autotune-v0/**/*','results/flashmoba-pool-stability-audit-v0/**/*',
            'logs/flashmoba-realtext-precision-*.log','logs/flashmoba-qwen*.log',
            'logs/flashmoba-fixed-pool*.log','logs/flashmoba-pool-autotune*.log',
            'configs/flashmoba-fixed-pool*.json','scripts/*flashmoba*pool*.py',
            'configs/flashmoba-qwen*.json','scripts/*flashmoba*precision*.py']:
    selected.update(p for p in ROOT.glob(pat) if p.is_file() and '__pycache__' not in p.parts)
for name in ['flashmoba-qwen-precision-v0','flashmoba-qwen-long-precision-v0']:
    folder=ROOT/'data'/name
    for p in folder.iterdir():
        if p.is_file():selected.add(p)
    if not (folder/'model').is_symlink():
        selected.update(p for p in (folder/'model').iterdir() if p.is_file() and p.suffix!='.safetensors')
for folder in [ROOT/'results/flashmoba-qwen-precision-v1',ROOT/'results/flashmoba-qwen-long-precision-v0']:
    assert json.loads((folder/'evaluation.json').read_text())['status']=='complete'
if version=='v1':assert (ROOT/'results/flashmoba-pool-stability-audit-v0/audit.json').exists()
mp=ROOT/f'provenance/flashmoba-precision-cloud-manifest-{version}.json';assert not mp.exists()
manifest=dict(utc=datetime.now(timezone.utc).isoformat(),scope='Raw precision experiments, failed gates and input metadata',
    excludes_public_model_safetensors=True,excluded_model_source='data/flashmoba-qwen-precision-v0/manifest.json',
    long_model_restore='Create data/flashmoba-qwen-long-precision-v0/model as a relative symlink to ../flashmoba-qwen-precision-v0/model on Linux.',
    files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(selected)])
mp.write_text(json.dumps(manifest,indent=2))
archive=Path(f'/workspace/flashmoba-precision-cloud-results-{version}.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(selected|{mp}):tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
proof=dict(utc=datetime.now(timezone.utc).isoformat(),archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,members=len(selected)+1)
(ROOT/f'provenance/flashmoba-precision-cloud-export-{version}.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof))
