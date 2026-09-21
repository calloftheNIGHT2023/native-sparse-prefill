#!/usr/bin/env bash
set -euo pipefail
cd /workspace/native-sparse-prefill
.venv-cloud/bin/python - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('results/cloud-dense-context-v0/result.json').read_text())
assert r['status']=='complete' and not r['fixed_context_gate']['passed']
c=json.loads(Path('configs/cloud-dense-context-10m-v0.json').read_text())
assert c['updates']*c['sequence_length']<=10000000
assert not Path('results/cloud-dense-context-10m-v0').exists()
print('1M context gate failed; bounded 10M control allowed by frozen next-stage plan')
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup timeout --signal=TERM --kill-after=30s 2100s .venv-cloud/bin/python -u src/run_cloud_control.py --config configs/cloud-dense-context-10m-v0.json --data data/cloud-articles-v2 --output results/cloud-dense-context-10m-v0 --device cuda --max-seconds 1800 > logs/cloud-dense-context-10m-v0.stdout.log 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > logs/cloud-dense-context-10m-v0.pid
printf 'Started bounded 10M control, PID %s\n' "$pid"
