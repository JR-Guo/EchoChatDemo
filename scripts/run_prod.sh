#!/usr/bin/env bash
# scripts/run_prod.sh — start echochat production server.
#
# Uses tmux for log visibility, but wraps gunicorn in `setsid` so it runs in
# its own session / process group. This makes it immune to SIGWINCH from
# the tmux pane being resized/closed (which was killing workers previously).
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
# Default: platformpah (eez194). Override with CONDA_PY env if needed.
CONDA_PY="${CONDA_PY:-/home/jguoaz/anaconda3/envs/platformpah/bin/python}"
HOST="${ECHOCHAT_HOST:-0.0.0.0}"
PORT="${ECHOCHAT_PORT:-12345}"

# Refuse to start if port already bound (another process is holding it).
if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
  echo "ERROR: port $PORT already in use. Free it first:" >&2
  echo "  tmux kill-session -t $SESSION 2>/dev/null" >&2
  echo "  pkill -9 -f 'gunicorn.*$PORT' 2>/dev/null" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already running. Stop it first with: tmux kill-session -t $SESSION"
  exit 1
fi

# Build the server command (single quoted, runs inside new session)
SERVER_CMD="export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'; \
export VIDEO_MAX_PIXELS='${VIDEO_MAX_PIXELS}'; \
export VIDEO_MIN_PIXELS='${VIDEO_MIN_PIXELS}'; \
export FPS='${FPS}'; \
export FPS_MAX_FRAMES='${FPS_MAX_FRAMES}'; \
export ECHOCHAT_ATTN_IMPL='${ECHOCHAT_ATTN_IMPL}'; \
set -a; . ./.env; set +a; \
export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'; \
export ECHOCHAT_ATTN_IMPL='${ECHOCHAT_ATTN_IMPL}'; \
exec '$CONDA_PY' -m gunicorn -k uvicorn.workers.UvicornWorker \
     -b $HOST:$PORT --workers 1 --timeout 600 \
     app.main:app 2>&1 | tee -a '$LOG_FILE'"

# `setsid` detaches from the tmux pane's terminal session, so SIGWINCH
# (pane resize) no longer reaches gunicorn.
tmux new-session -dA -s "$SESSION" \
  "setsid bash -c \"$SERVER_CMD\""

echo "Started tmux session '$SESSION' (cuda=${CUDA_VISIBLE_DEVICES} attn=${ECHOCHAT_ATTN_IMPL} py=${CONDA_PY})."
echo "  healthz:  curl http://127.0.0.1:$PORT/healthz"
echo "  logs:     tail -f $LOG_FILE"
echo "  attach:   tmux attach -t $SESSION   (Ctrl+B D to detach)"
