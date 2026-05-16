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

LOCAL_CIFAR="${REPO_ROOT}/data/cifar-10-python.tar.gz"
if [[ ! -f "${LOCAL_CIFAR}" ]]; then
  echo "Missing local CIFAR-10 archive: ${LOCAL_CIFAR}" >&2
  exit 1
fi

TARGET="${REMOTE_BJB3_USER}@${REMOTE_BJB3_HOST}"
SSH_OPTS=(-p "${REMOTE_BJB3_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_known_hosts_bjb3)

expect <<EOF
set timeout -1
spawn ssh ${SSH_OPTS[*]} ${TARGET} "mkdir -p '${REMOTE_BJB3_PROJECT_DIR}/data'"
expect {
  "*assword:*" { send "${REMOTE_BJB3_PASSWORD}\r"; exp_continue }
  eof
}
catch wait result
if {[lindex \$result 3] != 0} { exit [lindex \$result 3] }
spawn rsync -a --partial --inplace --progress -e "ssh ${SSH_OPTS[*]}" "${LOCAL_CIFAR}" "${TARGET}:${REMOTE_BJB3_PROJECT_DIR}/data/cifar-10-python.tar.gz"
expect {
  "*assword:*" { send "${REMOTE_BJB3_PASSWORD}\r"; exp_continue }
  eof
}
catch wait result
exit [lindex \$result 3]
EOF

echo "bjb3 CIFAR-10 archive deployed: ${TARGET}:${REMOTE_BJB3_PROJECT_DIR}/data/cifar-10-python.tar.gz"
