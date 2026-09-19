#!/usr/bin/env bash
set -uo pipefail
cd /workspace/native-sparse-prefill/schedule-study-v0
date -u +%Y-%m-%dT%H:%M:%SZ > logs/science-start.utc
/opt/sparse-schedule-env/bin/pip freeze > logs/clean-env-freeze.txt
timeout --signal=TERM --kill-after=30s 1800s /opt/sparse-schedule-env/bin/python -u src/run_sparse_schedule.py --output results/schedule-screen-v0 --device cuda --epochs 40 --max-seconds 600
status=$?
printf '%s\n' "$status" > logs/science-exit-code.txt
date -u +%Y-%m-%dT%H:%M:%SZ > logs/science-end.utc
exit "$status"
