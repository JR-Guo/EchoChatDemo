#!/usr/bin/env bash
# scripts/run_prod.sh — start the server under tmux session 'echochat-demo'.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

SESSION="echochat-demo"
LOG_DIR="./logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/service.log"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-20971520}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-13107}"
FPS="${FPS:-1}"
FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
ECHOCHAT_ATTN_IMPL="${ECHOCHAT_ATTN_IMPL:-sdpa}"
# Prefer platformpah (eez194) as default; override with CONDA_PY env if needed.
CONDA_PY="${CONDA_PY:-/home/jguoaz/anaconda3/envs/platformpah/bin/python}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already running. Stop it first with: tmux kill-session -t $SESSION"
  exit 1
fi

tmux new-session -dA -s "$SESSION" \
  "export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'; \
   export VIDEO_MAX_PIXELS='${VIDEO_MAX_PIXELS}'; \
   export VIDEO_MIN_PIXELS='${VIDEO_MIN_PIXELS}'; \
   export FPS='${FPS}'; \
   export FPS_MAX_FRAMES='${FPS_MAX_FRAMES}'; \
   export ECHOCHAT_ATTN_IMPL='${ECHOCHAT_ATTN_IMPL}'; \
   set -a; . ./.env; set +a; \
   export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'; \
   export ECHOCHAT_ATTN_IMPL='${ECHOCHAT_ATTN_IMPL}'; \
   exec '$CONDA_PY' -m gunicorn -k uvicorn.workers.UvicornWorker \
       -b \${ECHOCHAT_HOST:-0.0.0.0}:\${ECHOCHAT_PORT:-12345} \
       --workers 1 --timeout 600 \
       app.main:app 2>&1 | tee -a '$LOG_FILE'"

echo "Started tmux session '$SESSION' (cuda=${CUDA_VISIBLE_DEVICES} attn=${ECHOCHAT_ATTN_IMPL} py=${CONDA_PY}). Tail log: tail -f $LOG_FILE"
