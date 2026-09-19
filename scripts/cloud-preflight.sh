#!/usr/bin/env bash
set -euo pipefail
cd /workspace/native-sparse-prefill
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv-cloud/bin/python -m unittest discover -s tests > logs/cloud-correctness-tests.log 2>&1
for length in 256 512 1024 2048; do
  timeout --signal=TERM --kill-after=15s 150s .venv-cloud/bin/python src/benchmark_joint_device.py --device cuda --length "$length" --max-seconds 60 --output "logs/cuda-preflight-${length}.json" > "logs/cuda-preflight-${length}.log" 2>&1
done
