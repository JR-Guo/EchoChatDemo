#!/usr/bin/env bash
# scripts/smoke.sh — end-to-end smoke test against a running server.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-12345}"
BASE="http://${HOST}:${PORT}"
DICOM="${1:?usage: $0 /path/to/file.dcm}"
PASSWORD="${SHARED_PASSWORD:-echochat}"

COOKIE="$(mktemp)"
trap 'rm -f "$COOKIE"' EXIT

echo "1) Login"
curl -s -c "$COOKIE" -d "password=${PASSWORD}" -o /dev/null -w "%{http_code}\n" \
  "${BASE}/login"

echo "2) Create study"
SID=$(curl -s -b "$COOKIE" -X POST "${BASE}/api/study" | jq -r .study_id)
echo "   study_id=$SID"

echo "3) Upload DICOM"
curl -s -b "$COOKIE" -F "file=@${DICOM}" "${BASE}/api/study/${SID}/upload" | jq .

echo "4) Process (SSE, first 40 events)"
curl -s -b "$COOKIE" -N "${BASE}/api/study/${SID}/process" | head -n 40

echo "5) Study state"
curl -s -b "$COOKIE" "${BASE}/api/study/${SID}" | jq '{clips: .clips[].view, tasks: .tasks}'

echo "6) Start report"
TID=$(curl -s -b "$COOKIE" -X POST -H 'Content-Type: application/json' \
  -d '{}' "${BASE}/api/study/${SID}/task/report" | jq -r .task_id)
echo "   task_id=$TID"

echo "7) Stream task (first 60 events)"
curl -s -b "$COOKIE" -N "${BASE}/api/task/${TID}/stream" | head -n 60

echo "8) Final result"
curl -s -b "$COOKIE" "${BASE}/api/task/${TID}" | jq '.status, .sections[0]'

echo "9) Export PDF"
curl -s -b "$COOKIE" -o /tmp/report.pdf "${BASE}/api/study/${SID}/report/export?format=pdf"
echo "   saved /tmp/report.pdf ($(stat -c %s /tmp/report.pdf) bytes)"
