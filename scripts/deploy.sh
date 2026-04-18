#!/usr/bin/env bash
# scripts/deploy.sh — sync a local checkout to the eez195 deployment target.
# Usage: bash scripts/deploy.sh [user@host]
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${1:-jguoaz@eez195.ece.ust.hk}"
REMOTE_PATH="/nfs/usrhome2/EchoChatDemo"

rsync -av --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.mypy_cache' \
  --exclude '.ruff_cache' \
  --exclude 'node_modules' \
  --exclude 'data' \
  --exclude 'logs' \
  --exclude '*.log' \
  --exclude 'venv' \
  --exclude '.venv' \
  ./ "${REMOTE}:${REMOTE_PATH}/"

echo "Deployed to ${REMOTE}:${REMOTE_PATH}"
echo "On the server, run:   bash scripts/run_prod.sh"
