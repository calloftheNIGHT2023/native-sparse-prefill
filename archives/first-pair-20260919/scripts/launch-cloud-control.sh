#!/usr/bin/env bash
set -euo pipefail
cd /workspace/native-sparse-prefill
test -f logs/cuda-preflight-2048.json
test -f logs/cloud-cuda-correctness.log
test ! -e results/cloud-dense-context-v0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup timeout --signal=TERM --kill-after=30s 3900s .venv-cloud/bin/python -u src/run_cloud_control.py --config configs/cloud-dense-context-v0.json --data data/cloud-articles-v2 --output results/cloud-dense-context-v0 --device cuda --max-seconds 3600 > logs/cloud-dense-context-v0.stdout.log 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > logs/cloud-dense-context-v0.pid
printf 'Started bounded cloud control, PID %s\n' "$pid"
