"""Extract a small fixed-input backward check kit; no checkpoint or training data download needed."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile
import torch
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=ROOT/'data/flashmoba-backward-fixed-v0';out.mkdir(parents=True,exist_ok=False)
shutil.copy2(ROOT/'results/flashmoba-qwen-long-precision-v0/fixed-real-qkv-example.pt',out/'qkv.pt')
go=torch.load(ROOT/'results/flashmoba-pool-repair-v1/gradient-repeats.pt',weights_only=True)['upstream_gradient']
torch.save(go,out/'upstream-gradient.pt')
means=torch.load(ROOT/'results/flashmoba-pool-autotune-v0/pooled-means-by-config.pt',weights_only=True)['torch.float32_bn32_w4_s3']
torch.save(means,out/'pooled-means.pt')
for name in ['fp64-reference.pt','reconstructed-routes.pt']:
    shutil.copy2(ROOT/'results/flashmoba-backward-cpu-reference-v1'/name,out/name)
manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Frozen QKV, upstream gradient, pooled keys and conditional FP64 reference. No model weights; no optimizer updates; do not retrain. Capture actual K4 mask before claiming gradient correctness.',
    upstream_commit='39d9ac043b271d046a2181a9991e99a26b67bca1',original_extension_sha256='b114a7755aad6556bc72eac1f8de0fcd0d5bd1e78d0dc1b1e521ac8621ce4d53',
    files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(directory=str(out),bytes=sum(p.stat().st_size for p in out.iterdir()))))
