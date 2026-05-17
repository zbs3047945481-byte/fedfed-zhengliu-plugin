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

RUN_ID="cross_domain_5090_$(date -u +%Y%m%dT%H%M%SZ)"
SUITE_PATH="experiments/cross_domain_default_a0p1_e1_5090.json"
MAX_WORKERS="2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --suite)
      SUITE_PATH="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

"${SCRIPT_DIR}/deploy_to_5090.sh"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/codex-5090-cross-domain.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

WORKER_FILE="${TMP_DIR}/${RUN_ID}_worker.sh"
REMOTE_WORKER_PATH="${REMOTE_5090_CODEX_DIR}/${RUN_ID}_worker.sh"
TARGET="${REMOTE_5090_USER}@${REMOTE_5090_HOST}"
SSH_BASE="ssh -p ${REMOTE_5090_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_known_hosts_5090"

cat > "${WORKER_FILE}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR='${REMOTE_5090_PROJECT_DIR}'
RUNS_DIR='${REMOTE_5090_RUNS_DIR}'
PYTHON='${REMOTE_5090_PYTHON}'
RUN_ID='${RUN_ID}'
SUITE_PATH='${SUITE_PATH}'
MAX_WORKERS='${MAX_WORKERS}'
RUN_DIR="\${RUNS_DIR}/\${RUN_ID}"
STATUS_PATH="\${RUN_DIR}/run_status.json"
STDOUT_PATH="\${RUN_DIR}/stdout.log"
STDERR_PATH="\${RUN_DIR}/stderr.log"

mkdir -p "\${RUN_DIR}" /tmp/matplotlib-cache
"\${PYTHON}" - <<PY
import json, pathlib, datetime
path = pathlib.Path("\${STATUS_PATH}")
path.write_text(json.dumps({
    "run_id": "${RUN_ID}",
    "status": "preparing_data",
    "started_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "suite": "${SUITE_PATH}",
    "stdout": "\${STDOUT_PATH}",
    "stderr": "\${STDERR_PATH}",
}, indent=2), encoding="utf-8")
PY

cd "\${PROJECT_DIR}"
export MPLCONFIGDIR=/tmp/matplotlib-cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

{
  echo "Preparing datasets at \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "\${PYTHON}" -u scripts/prepare_cross_domain_datasets.py \
    --datasets gtsrb pathmnist plantvillage \
    --data-root ./data \
    --image-size 32 \
    --seed 3001
  echo "Starting suite at \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "\${STDOUT_PATH}" 2> "\${STDERR_PATH}"

"\${PYTHON}" - <<PY
import json, pathlib, datetime
path = pathlib.Path("\${STATUS_PATH}")
path.write_text(json.dumps({
    "run_id": "${RUN_ID}",
    "status": "running",
    "started_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "suite": "${SUITE_PATH}",
    "stdout": "\${STDOUT_PATH}",
    "stderr": "\${STDERR_PATH}",
}, indent=2), encoding="utf-8")
PY

set +e
"\${PYTHON}" -u scripts/run_linux_thesis_suite.py \
  --suite "\${SUITE_PATH}" \
  --project-dir "\${PROJECT_DIR}" \
  --runs-root "\${RUNS_DIR}" \
  --prefix "\${RUN_ID}" \
  --python "\${PYTHON}" \
  --max-workers "\${MAX_WORKERS}" \
  --skip-groups "" \
  >> "\${STDOUT_PATH}" 2>> "\${STDERR_PATH}"
EXIT_CODE=\$?
set -e

"\${PYTHON}" - <<PY
import json, pathlib, datetime
exit_code = \${EXIT_CODE}
path = pathlib.Path("\${STATUS_PATH}")
path.write_text(json.dumps({
    "run_id": "${RUN_ID}",
    "status": "succeeded" if exit_code == 0 else "failed",
    "exit_code": exit_code,
    "finished_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "queue_status": "\${RUNS_DIR}/${RUN_ID}_queue_status.json",
    "summary_json": "\${RUNS_DIR}/${RUN_ID}_summary.json",
    "summary_csv": "\${RUNS_DIR}/${RUN_ID}_summary.csv",
    "figures_dir": "\${RUNS_DIR}/${RUN_ID}_figures",
    "stdout": "\${STDOUT_PATH}",
    "stderr": "\${STDERR_PATH}",
}, indent=2), encoding="utf-8")
PY
exit "\${EXIT_CODE}"
EOF

expect <<EOF
set timeout -1
spawn ${SSH_BASE} ${TARGET} "mkdir -p '${REMOTE_5090_CODEX_DIR}' '${REMOTE_5090_RUNS_DIR}'"
expect {
  "*assword:*" { send "${REMOTE_5090_PASSWORD}\r"; exp_continue }
  eof
}
catch wait result
exit [lindex \$result 3]
EOF

expect <<EOF
set timeout -1
spawn scp -P ${REMOTE_5090_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_known_hosts_5090 "${WORKER_FILE}" "${TARGET}:${REMOTE_WORKER_PATH}"
expect {
  "*assword:*" { send "${REMOTE_5090_PASSWORD}\r"; exp_continue }
  eof
}
catch wait result
exit [lindex \$result 3]
EOF

expect <<EOF
set timeout -1
spawn ${SSH_BASE} ${TARGET} "chmod +x '${REMOTE_WORKER_PATH}' && nohup bash '${REMOTE_WORKER_PATH}' > '${REMOTE_5090_RUNS_DIR}/${RUN_ID}_launcher.log' 2>&1 & echo STARTED"
expect {
  "*assword:*" { send "${REMOTE_5090_PASSWORD}\r"; exp_continue }
  eof
}
catch wait result
exit [lindex \$result 3]
EOF

echo "5090 cross-domain suite submitted: ${RUN_ID}"
echo "Check: REMOTE_5090_PASSWORD=... ./scripts/check_remote_5090_run.sh --run-id ${RUN_ID}"
