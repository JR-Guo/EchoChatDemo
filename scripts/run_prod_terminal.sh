#!/usr/bin/env bash
# scripts/run_prod_terminal.sh — start the server in the current terminal.
#
# Everything the server needs comes from `.env` (ECHOCHAT_MODEL_PATH,
# ECHOCHAT_DATA_DIR, SHARED_PASSWORD, SESSION_SECRET, CUDA_VISIBLE_DEVICES, …).
# Two vars can be overridden at invocation time:
#
#   CONDA_PY        absolute path to a python with ms-swift / torch / pyjwt /
#                   SimpleITK / decord / opencv-python-headless installed
#                   (defaults to a common location below — override if yours differs).
#
#   ECHOCHAT_ATTN_IMPL  sdpa (default) or flash_attn
#
# Example:
#   CONDA_PY=/home/jguoaz/anaconda3/envs/qwen/bin/python bash scripts/run_prod_terminal.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  . ./.env
  set +a
fi

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/service.log"

# Non-critical tuning defaults (can also live in .env)
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-20971520}"
export VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-13107}"
export FPS="${FPS:-1}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
export ECHOCHAT_ATTN_IMPL="${ECHOCHAT_ATTN_IMPL:-sdpa}"

# Python env. Try a couple of common locations so `bash scripts/run_prod_terminal.sh`
# works without any explicit override on most of our machines.
_candidate_pys=(
  "${CONDA_PY:-}"
  "/home/jguoaz/anaconda3/envs/qwen/bin/python"
  "/home/yqinar/miniconda3/envs/echochat/bin/python"
)
CONDA_PY=""
for p in "${_candidate_pys[@]}"; do
  [[ -z "$p" ]] && continue
  if [[ -x "$p" ]]; then
    CONDA_PY="$p"
    break
  fi
done
if [[ -z "$CONDA_PY" ]]; then
  echo "ERROR: no suitable python found. Set CONDA_PY to an env with ms-swift + pyjwt + SimpleITK + decord." >&2
  exit 1
fi

# Required: model + data come from .env; fail fast if missing so we don't
# silently use someone else's weights.
: "${ECHOCHAT_MODEL_PATH:?ECHOCHAT_MODEL_PATH must be set in .env}"
: "${ECHOCHAT_DATA_DIR:?ECHOCHAT_DATA_DIR must be set in .env}"
: "${SHARED_PASSWORD:?SHARED_PASSWORD must be set in .env}"
: "${SESSION_SECRET:?SESSION_SECRET must be set in .env}"

if [[ ! -d "$ECHOCHAT_MODEL_PATH" ]]; then
  echo "ERROR: ECHOCHAT_MODEL_PATH ($ECHOCHAT_MODEL_PATH) does not exist or is unreadable." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

echo "Starting server (cuda=${CUDA_VISIBLE_DEVICES} attn=${ECHOCHAT_ATTN_IMPL}"
echo "                 py=${CONDA_PY}"
echo "                 model=${ECHOCHAT_MODEL_PATH})"

exec "$CONDA_PY" -m gunicorn -k uvicorn.workers.UvicornWorker \
  -b "${ECHOCHAT_HOST:-0.0.0.0}:${ECHOCHAT_PORT:-12345}" \
  --workers 1 --timeout 600 \
  app.main:app 2>&1 | tee -a "$LOG_FILE"
