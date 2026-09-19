"""Pin public implementation sources for a read-only fidelity audit; execute none."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parents[1]
out = root / 'literature/indexer-fidelity-2026-09-13'
out.mkdir(exist_ok=False)
nemo = 'https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/f7ccd6f7902634af34c2f31b3294ac250dc97670/'
vllm = 'https://raw.githubusercontent.com/vllm-project/vllm/319cc5ef19946d34c2e66cbbec5bda29d0bfa328/'
sources = {
    'nemo-qsa.py': nemo + 'nemo_automodel/components/models/qwen3_8_flash_next/qsa.py',
    'nemo-layers.py': nemo + 'nemo_automodel/components/models/qwen3_8_flash_next/layers.py',
    'vllm-indexer.py': vllm + 'vllm/models/qwen4_exp/nvidia/indexer_qsa.py',
    'vllm-pre-indexer.py': vllm + 'vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py',
    'vllm-indexer-kernel.py': vllm + 'vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py',
    'qwen-config.json': 'https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json',
}
manifest = []
for name, url in sources.items():
    started = datetime.now(timezone.utc).isoformat()
    data = urllib.request.urlopen(url, timeout=45).read()
    (out / name).write_bytes(data)
    manifest.append({'path': name, 'url': url, 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest()})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'files_saved': len(manifest), 'directory': str(out)}))
