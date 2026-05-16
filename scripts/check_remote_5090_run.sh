#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_5090_ENV_FILE="${REMOTE_5090_ENV_FILE:-${SCRIPT_DIR}/remote_5090.env}"

# shellcheck disable=SC1090
source "${REMOTE_5090_ENV_FILE}"

if [[ -z "${REMOTE_5090_PASSWORD:-}" ]]; then
  echo "Missing REMOTE_5090_PASSWORD. Export it for this shell; do not save it in files." >&2
  exit 1
fi

RUN_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 --run-id <run_id>" >&2
  exit 1
fi

TARGET="${REMOTE_5090_USER}@${REMOTE_5090_HOST}"

expect <<EOF
set timeout 60
spawn ssh -p ${REMOTE_5090_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_known_hosts_5090 ${TARGET} bash -lc {echo STATUS_FILE=${REMOTE_5090_RUNS_DIR}/${RUN_ID}/run_status.json; cat ${REMOTE_5090_RUNS_DIR}/${RUN_ID}/run_status.json 2>/dev/null || echo missing_status; echo '--- GPU ---'; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; echo '--- TAIL STDOUT ---'; tail -40 ${REMOTE_5090_RUNS_DIR}/${RUN_ID}/stdout.log 2>/dev/null || true; echo '--- TAIL STDERR ---'; tail -40 ${REMOTE_5090_RUNS_DIR}/${RUN_ID}/stderr.log 2>/dev/null || true}
expect {
  "*assword:*" { send "${REMOTE_5090_PASSWORD}\r"; exp_continue }
  eof
}
catch wait result
exit [lindex \$result 3]
EOF
