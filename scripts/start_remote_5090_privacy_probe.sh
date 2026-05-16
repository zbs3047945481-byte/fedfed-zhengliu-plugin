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

RUN_ID="fedfed_privacy_5090_$(date -u +%Y%m%dT%H%M%SZ)"
VARIANTS="A0_noclip_none:none:0.0:0.0:false:none,A1_noclip_g020_025:gaussian:0.20:0.25:true:none,C14_g020_025:gaussian:0.20:0.25:true:14,C12_g020_025:gaussian:0.20:0.25:true:12,C10_g020_025:gaussian:0.20:0.25:true:10,C12_g010_015:gaussian:0.10:0.15:true:12,C12_g015_020:gaussian:0.15:0.20:true:12,C12_g025_030:gaussian:0.25:0.30:true:12,C14_g015_020:gaussian:0.15:0.20:true:14,C10_g015_020:gaussian:0.15:0.20:true:10"
SEED="${FEDFED_PRIVACY_SEED:-3001}"
NOISE_SHAPE="${FEDFED_NOISE_SHAPE:-paper}"
SHADOW_MODELS="${FEDFED_SHADOW_MODELS:-2}"
SHADOW_TRAIN_SAMPLES="${FEDFED_SHADOW_TRAIN_SAMPLES:-400}"
SHADOW_TEST_SAMPLES="${FEDFED_SHADOW_TEST_SAMPLES:-400}"
MIA_SAMPLES="${FEDFED_MIA_SAMPLES:-800}"
INVERSION_EPOCHS="${FEDFED_INVERSION_EPOCHS:-4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --variants)
      VARIANTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

"${SCRIPT_DIR}/deploy_to_5090.sh"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/codex-5090.XXXXXX")"
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
RUN_DIR="\${RUNS_DIR}/\${RUN_ID}"
STATUS_PATH="\${RUN_DIR}/run_status.json"
STDOUT_PATH="\${RUN_DIR}/stdout.log"
STDERR_PATH="\${RUN_DIR}/stderr.log"
RESULT_DIR="\${PROJECT_DIR}/result/cifar10/fedfed_privacy_probe_\${RUN_ID}"

mkdir -p "\${RUN_DIR}"
"\${PYTHON}" - <<PY
import json, pathlib, datetime
path = pathlib.Path("\${STATUS_PATH}")
path.write_text(json.dumps({
    "run_id": "${RUN_ID}",
    "status": "running",
    "started_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "variants": "${VARIANTS}",
    "stdout": "\${STDOUT_PATH}",
    "stderr": "\${STDERR_PATH}",
}, indent=2), encoding="utf-8")
PY

cd "\${PROJECT_DIR}"
export FEDFED_PRIVACY_VARIANTS='${VARIANTS}'

set +e
"\${PYTHON}" -u scripts/run_fedfed_privacy_probe.py \
  --dataset_name cifar10 \
  --model_name cifar_resnet18 \
  --round_num 1 \
  --num_of_clients 10 \
  --c_fraction 0.5 \
  --local_epoch 1 \
  --batch_size 64 \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory true \
  --torch_cudnn_benchmark true \
  --gpu true \
  --seed '${SEED}' \
  --partition_strategy dirichlet \
  --dirichlet_alpha 0.1 \
  --min_samples_per_client 32 \
  --enable_quantity_skew false \
  --enable_feature_skew false \
  --lr 0.01 \
  --optimizer_name sgd \
  --weight_decay 0.0001 \
  --momentum 0.9 \
  --plugin_name fedfed_image \
  --fedfed_generator_type paper_beta_vae \
  --fedfed_vae_latent_channels 32 \
  --fedfed_vae_z_dim 2048 \
  --fedfed_distill_rounds 15 \
  --fedfed_distill_local_epoch 1 \
  --fedfed_distill_optimizer adamw \
  --fedfed_distill_lr 0.001 \
  --fedfed_distill_weight_decay 0.000001 \
  --fedfed_lambda_recon 0.2 \
  --fedfed_lambda_fd 1.5 \
  --fedfed_beta_kl 0.005 \
  --fedfed_lambda_x_ce 0.4 \
  --fedfed_lambda_logit_align 0.5 \
  --fedfed_logit_align_temperature 2.0 \
  --fedfed_use_augmentation true \
  --fedfed_mixup_alpha 2.0 \
  --fedfed_mosaic_batch_size 64 \
  --fedfed_shared_resident_device cuda \
  --fedfed_shared_cpu_float16 false \
  --fedfed_noise_shape '${NOISE_SHAPE}' \
  --privacy_shadow_models '${SHADOW_MODELS}' \
  --privacy_shadow_train_samples '${SHADOW_TRAIN_SAMPLES}' \
  --privacy_shadow_test_samples '${SHADOW_TEST_SAMPLES}' \
  --privacy_mia_samples '${MIA_SAMPLES}' \
  --privacy_inversion_epochs '${INVERSION_EPOCHS}' \
  --diagnostic_epochs 6 \
  --diagnostic_train_limit 12000 \
  --diagnostic_test_limit 4000 \
  --experiment_tag "\${RUN_ID}" \
  > "\${STDOUT_PATH}" 2> "\${STDERR_PATH}"
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
    "result_dir": "\${RESULT_DIR}",
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

echo "5090 privacy probe submitted: ${RUN_ID}"
echo "Check: REMOTE_5090_PASSWORD=... ./scripts/check_remote_5090_run.sh --run-id ${RUN_ID}"
