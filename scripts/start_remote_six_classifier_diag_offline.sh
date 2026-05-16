#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

RUN_ID="fedfed_six_classifier_diag_$(date -u +%Y%m%dT%H%M%SZ)"
TASK_NAME="FedFedSixDiag_${RUN_ID}"
TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT

BAT_FILE="${TMP_DIR}/six_classifier_diag_${RUN_ID}.bat"
REMOTE_BAT_PATH="${REMOTE_Codex_DIR}/six_classifier_diag_${RUN_ID}.bat"
REMOTE_BAT_WIN_PATH="${REMOTE_BAT_PATH//\//\\}"

"${SCRIPT_DIR}/deploy_to_windows.sh" >/dev/null

cat > "${BAT_FILE}" <<EOF
@echo off
set RUN_ID=${RUN_ID}
set RUN_DIR=${REMOTE_RUNS_DIR}\\${RUN_ID}
set PROJECT_DIR=${REMOTE_PROJECT_DIR}
set PYTHON_EXE=${REMOTE_PYTHON}

if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"
echo {"run_id":"%RUN_ID%","status":"running","kind":"six_classifier_diagnostic","stdout":"%RUN_DIR%\\\\stdout.log","stderr":"%RUN_DIR%\\\\stderr.log"} > "%RUN_DIR%\\run_status.json"

cd /d "%PROJECT_DIR%"
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
"%PYTHON_EXE%" -u scripts\\run_fedfed_six_classifier_diagnostic.py ^
  --dataset_name cifar10 ^
  --model_name cifar_resnet18 ^
  --round_num 1 ^
  --num_of_clients 10 ^
  --c_fraction 0.5 ^
  --local_epoch 1 ^
  --batch_size 64 ^
  --dataloader_num_workers 0 ^
  --dataloader_pin_memory true ^
  --torch_cudnn_benchmark true ^
  --client_empty_cache false ^
  --gpu true ^
  --partition_strategy dirichlet ^
  --dirichlet_alpha 0.1 ^
  --min_samples_per_client 1 ^
  --enable_quantity_skew true ^
  --enable_feature_skew false ^
  --optimizer_name sgd ^
  --lr 0.01 ^
  --weight_decay 0.0001 ^
  --momentum 0.9 ^
  --nesterov false ^
  --lr_schedule none ^
  --plugin_name fedfed_image ^
  --fedfed_two_stage true ^
  --fedfed_generator_type paper_beta_vae ^
  --fedfed_vae_latent_channels 32 ^
  --fedfed_vae_z_dim 2048 ^
  --fedfed_distill_rounds 15 ^
  --fedfed_distill_local_epoch 1 ^
  --fedfed_distill_optimizer adamw ^
  --fedfed_distill_lr 0.001 ^
  --fedfed_distill_weight_decay 0.000001 ^
  --fedfed_lambda_recon 5.0 ^
  --fedfed_lambda_fd 2.0 ^
  --fedfed_beta_kl 0.005 ^
  --fedfed_lambda_x_ce 0.4 ^
  --fedfed_use_augmentation true ^
  --fedfed_mixup_alpha 2.0 ^
  --fedfed_mosaic_batch_size 32 ^
  --fedfed_upload_per_class 0 ^
  --fedfed_upload_per_client 0 ^
  --fedfed_shared_buffer_size 0 ^
  --fedfed_shared_per_class_size 0 ^
  --fedfed_shared_batch_size 0 ^
  --fedfed_shared_cpu_float16 false ^
  --fedfed_shared_resident_device cuda ^
  --fedfed_collapse_duplicate_shared_no_noise true ^
  --fedfed_noise_type gaussian ^
  --fedfed_noise_mean 0.0 ^
  --fedfed_noise_std1 0.2 ^
  --fedfed_noise_std2 0.0 ^
  --fedfed_distill_ce_on_noisy_xs false ^
  --diagnostic_epochs 10 ^
  --diagnostic_train_limit 20000 ^
  --diagnostic_test_limit 10000 ^
  --experiment_tag "%RUN_ID%" > "%RUN_DIR%\\stdout.log" 2> "%RUN_DIR%\\stderr.log"

if errorlevel 1 (
  echo {"run_id":"%RUN_ID%","status":"failed","kind":"six_classifier_diagnostic","stdout":"%RUN_DIR%\\\\stdout.log","stderr":"%RUN_DIR%\\\\stderr.log"} > "%RUN_DIR%\\run_status.json"
  exit /b 1
)

echo {"run_id":"%RUN_ID%","status":"succeeded","kind":"six_classifier_diagnostic","stdout":"%RUN_DIR%\\\\stdout.log","stderr":"%RUN_DIR%\\\\stderr.log","metrics_path":"%PROJECT_DIR%\\\\result\\\\cifar10\\\\fedfed_six_classifier_diagnostic_%RUN_ID%\\\\diagnostic_metrics.json"} > "%RUN_DIR%\\run_status.json"
EOF

ssh_remote "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_Codex_DIR}' | Out-Null\""
scp_to_remote "${BAT_FILE}" "${REMOTE_BAT_PATH}"
ssh_remote "schtasks /Create /TN ${TASK_NAME} /SC ONCE /ST 23:59 /TR \"cmd.exe /c ${REMOTE_BAT_WIN_PATH}\" /F"
ssh_remote "schtasks /Run /TN ${TASK_NAME}"

echo "RUN_ID=${RUN_ID}"
echo "TASK_NAME=${TASK_NAME}"
echo "STATUS=${REMOTE_RUNS_DIR}\\${RUN_ID}\\run_status.json"
