#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_5090_ENV_FILE="${REMOTE_5090_ENV_FILE:-${SCRIPT_DIR}/remote_5090.env}"

# shellcheck disable=SC1090
source "${REMOTE_5090_ENV_FILE}"

if [[ -z "${REMOTE_5090_PASSWORD:-}" ]]; then
  echo "Missing REMOTE_5090_PASSWORD. Export it for this shell; do not save it in files." >&2
  exit 1
fi

SSH_OPTS=(-p "${REMOTE_5090_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_known_hosts_5090)
TARGET="${REMOTE_5090_USER}@${REMOTE_5090_HOST}"

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
  "${REPO_ROOT}/" "${TARGET}:${REMOTE_5090_PROJECT_DIR}/"
expect {
  "*assword:*" { send "${REMOTE_5090_PASSWORD}\r"; exp_continue }
  eof
}
catch wait result
exit [lindex \$result 3]
EOF

echo "5090 deployment finished: ${TARGET}:${REMOTE_5090_PROJECT_DIR}"
