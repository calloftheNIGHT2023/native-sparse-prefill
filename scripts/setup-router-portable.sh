#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="${SPARSE_RUNTIME_DIR:-/opt/sparse-router-env}"
BASE_PYTHON="${SPARSE_BASE_PYTHON:-python3}"
cd "$PROJECT_DIR"
mkdir -p logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
exec > >(tee -a "logs/migration-setup-$STAMP.log") 2>&1
"$BASE_PYTHON" -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12), "Use Python 3.11, or supported 3.10/3.12; this pinned runtime excludes 3.13+."; print(sys.version)'
# Never merge an unknown or damaged venv into this environment.
if [[ -e "$RUNTIME_DIR" ]]; then
  echo "Runtime already exists: $RUNTIME_DIR. Inspect it or choose a new SPARSE_RUNTIME_DIR; nothing was overwritten." >&2
  exit 2
fi
"$BASE_PYTHON" -m venv "$RUNTIME_DIR"
"$RUNTIME_DIR/bin/python" -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
"$RUNTIME_DIR/bin/python" -m pip install -r requirements-router-portable.txt
"$RUNTIME_DIR/bin/python" -m pip check
"$RUNTIME_DIR/bin/python" -m pip freeze > "logs/migration-runtime-freeze-$STAMP.txt"
"$RUNTIME_DIR/bin/python" scripts/verify-router-migration.py --device cuda --output "logs/migration-gpu-check-$STAMP.json"
echo "Environment and read-only checkpoint evaluation passed. No scientific training was started."
