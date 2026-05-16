#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_BJB3_ENV_FILE="${REMOTE_BJB3_ENV_FILE:-${SCRIPT_DIR}/remote_bjb3.env}"

# shellcheck disable=SC1090
source "${REMOTE_BJB3_ENV_FILE}"

if [[ -z "${REMOTE_BJB3_PASSWORD:-}" ]]; then
  echo "Missing REMOTE_BJB3_PASSWORD. Export it for this shell; do not save it in files." >&2
  exit 1
fi

SSH_OPTS=(-p "${REMOTE_BJB3_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_known_hosts_bjb3)
TARGET="${REMOTE_BJB3_USER}@${REMOTE_BJB3_HOST}"

expect <<EOF
set timeout -1
spawn rsync -az --delete \
  --exclude .git \
  --exclude __pycache__ \
  --exclude "*.pyc" \
  --exclude data \
  --exclude result \
  --exclude remote_results \
  --exclude outputs \
  --exclude tmp \
  --exclude .npm-cache \
  --exclude .DS_Store \
  -e "ssh ${SSH_OPTS[*]}" \
  "${REPO_ROOT}/" "${TARGET}:${REMOTE_BJB3_PROJECT_DIR}/"
expect {
  "*assword:*" { send "${REMOTE_BJB3_PASSWORD}\r"; exp_continue }
  eof
}
catch wait result
exit [lindex \$result 3]
EOF

echo "bjb3 deployment finished: ${TARGET}:${REMOTE_BJB3_PROJECT_DIR}"
