#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

RUN_ID="fedfed_six_classifier_diag_$(date -u +%Y%m%dT%H%M%SZ)"
TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT

WORKER_FILE="${TMP_DIR}/six_classifier_diag_${RUN_ID}.ps1"
LAUNCHER_FILE="${TMP_DIR}/six_classifier_diag_launcher_${RUN_ID}.ps1"
REMOTE_WORKER_PATH="${REMOTE_Codex_DIR}/six_classifier_diag_${RUN_ID}.ps1"
REMOTE_LAUNCHER_PATH="${REMOTE_Codex_DIR}/six_classifier_diag_launcher_${RUN_ID}.ps1"
REMOTE_WORKER_PS_PATH="${REMOTE_WORKER_PATH//\//\\}"
REMOTE_LAUNCHER_PS_PATH="${REMOTE_LAUNCHER_PATH//\//\\}"

cat > "${WORKER_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$pythonPath = '${REMOTE_PYTHON}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$runId = '${RUN_ID}'
\$runDir = Join-Path \$runsRoot \$runId
\$stdoutPath = Join-Path \$runDir 'stdout.log'
\$stderrPath = Join-Path \$runDir 'stderr.log'
\$statusPath = Join-Path \$runDir 'run_status.json'
\$summaryPath = Join-Path \$runDir 'run_summary.json'

New-Item -ItemType Directory -Force -Path \$runDir | Out-Null
[ordered]@{
  run_id = \$runId
  status = 'running'
  kind = 'six_classifier_diagnostic'
  started_utc = (Get-Date).ToUniversalTime().ToString('o')
  stdout = \$stdoutPath
  stderr = \$stderrPath
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath

try {
  Set-Location \$projectPath
  \$env:PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:128'
  \$args = @(
    '--dataset_name','cifar10',
    '--model_name','cifar_resnet18',
    '--round_num','1',
    '--num_of_clients','10',
    '--c_fraction','0.5',
    '--local_epoch','1',
    '--batch_size','64',
    '--dataloader_num_workers','0',
    '--dataloader_pin_memory','true',
    '--torch_cudnn_benchmark','true',
    '--client_empty_cache','false',
    '--gpu','true',
    '--partition_strategy','dirichlet',
    '--dirichlet_alpha','0.1',
    '--min_samples_per_client','1',
    '--enable_quantity_skew','true',
    '--enable_feature_skew','false',
    '--optimizer_name','sgd',
    '--lr','0.01',
    '--weight_decay','0.0001',
    '--momentum','0.9',
    '--nesterov','false',
    '--lr_schedule','none',
    '--plugin_name','fedfed_image',
    '--fedfed_two_stage','true',
    '--fedfed_generator_type','paper_beta_vae',
    '--fedfed_vae_latent_channels','32',
    '--fedfed_vae_z_dim','2048',
    '--fedfed_distill_rounds','15',
    '--fedfed_distill_local_epoch','1',
    '--fedfed_distill_optimizer','adamw',
    '--fedfed_distill_lr','0.001',
    '--fedfed_distill_weight_decay','0.000001',
    '--fedfed_lambda_recon','5.0',
    '--fedfed_lambda_fd','2.0',
    '--fedfed_beta_kl','0.005',
    '--fedfed_lambda_x_ce','0.4',
    '--fedfed_use_augmentation','true',
    '--fedfed_mixup_alpha','2.0',
    '--fedfed_mosaic_batch_size','32',
    '--fedfed_upload_per_class','0',
    '--fedfed_upload_per_client','0',
    '--fedfed_shared_buffer_size','0',
    '--fedfed_shared_per_class_size','0',
    '--fedfed_shared_batch_size','0',
    '--fedfed_shared_cpu_float16','false',
    '--fedfed_shared_resident_device','cuda',
    '--fedfed_collapse_duplicate_shared_no_noise','true',
    '--fedfed_noise_type','gaussian',
    '--fedfed_noise_mean','0.0',
    '--fedfed_noise_std1','0.2',
    '--fedfed_noise_std2','0.0',
    '--fedfed_distill_ce_on_noisy_xs','false',
    '--diagnostic_epochs','10',
    '--diagnostic_train_limit','20000',
    '--diagnostic_test_limit','10000',
    '--experiment_tag',\$runId
  )
  [ordered]@{
    run_id = \$runId
    project_path = \$projectPath
    python_path = \$pythonPath
    arguments = \$args
  } | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 \$summaryPath
  & \$pythonPath '-u' 'scripts\\run_fedfed_six_classifier_diagnostic.py' @args 1> \$stdoutPath 2> \$stderrPath
  if (\$LASTEXITCODE -ne 0) {
    throw "six-classifier diagnostic failed with exit code \$LASTEXITCODE"
  }
  \$resultDir = Join-Path \$projectPath ('result\\cifar10\\fedfed_six_classifier_diagnostic_' + \$runId)
  \$metricsPath = Join-Path \$resultDir 'diagnostic_metrics.json'
  [ordered]@{
    run_id = \$runId
    status = 'succeeded'
    kind = 'six_classifier_diagnostic'
    finished_utc = (Get-Date).ToUniversalTime().ToString('o')
    stdout = \$stdoutPath
    stderr = \$stderrPath
    result_dir = \$resultDir
    metrics_path = \$metricsPath
  } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath
} catch {
  [ordered]@{
    run_id = \$runId
    status = 'failed'
    kind = 'six_classifier_diagnostic'
    finished_utc = (Get-Date).ToUniversalTime().ToString('o')
    stdout = \$stdoutPath
    stderr = \$stderrPath
    error = (\$_ | Out-String)
  } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath
  throw
}
EOF

cat > "${LAUNCHER_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','${REMOTE_WORKER_PS_PATH}') -WindowStyle Hidden
EOF

scp_to_remote "${WORKER_FILE}" "${REMOTE_WORKER_PATH}"
scp_to_remote "${LAUNCHER_FILE}" "${REMOTE_LAUNCHER_PATH}"
ssh_remote "powershell -NoProfile -ExecutionPolicy Bypass -File ${REMOTE_LAUNCHER_PS_PATH}"

echo "RUN_ID=${RUN_ID}"
echo "STATUS=${REMOTE_RUNS_DIR}\\${RUN_ID}\\run_status.json"
