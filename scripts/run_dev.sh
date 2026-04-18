#!/usr/bin/env bash
# scripts/run_dev.sh — local hot-reload dev server.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export ECHOCHAT_SKIP_MODEL="${ECHOCHAT_SKIP_MODEL:-1}"
uvicorn app.main:app --reload --host "${ECHOCHAT_HOST:-127.0.0.1}" \
  --port "${ECHOCHAT_PORT:-12345}"
