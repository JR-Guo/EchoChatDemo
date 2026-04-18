#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export ECHOCHAT_SKIP_MODEL="${ECHOCHAT_SKIP_MODEL:-1}"
CONDA_PY="${CONDA_PY:-/home/jguoaz/anaconda3/envs/qwen/bin/python}"
exec "$CONDA_PY" -m uvicorn app.main:app --reload \
  --host "${ECHOCHAT_HOST:-127.0.0.1}" --port "${ECHOCHAT_PORT:-12345}"
