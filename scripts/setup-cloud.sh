#!/usr/bin/env bash
set -euo pipefail
cd /workspace/native-sparse-prefill
mkdir -p logs results
python -m venv --system-site-packages .venv-cloud
.venv-cloud/bin/python -m pip install 'transformers==4.57.6' 'pyarrow==22.0.0'
.venv-cloud/bin/python scripts/cloud-check-env.py
.venv-cloud/bin/python -m pip freeze > logs/cloud-environment-freeze.txt
.venv-cloud/bin/python scripts/cloud-inspect.py
