#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

RUN_ID="fedfed_privacy_probe_$(date -u +%Y%m%dT%H%M%SZ)"
VARIANTS="none:none:0.0:0.0:false,g010_015:gaussian:0.10:0.15:true,g015_020:gaussian:0.15:0.20:true,g020_025:gaussian:0.20:0.25:true"
CLIP_NORM="${FEDFED_CLIP_NORM:-0.0}"
NOISE_SHAPE="${FEDFED_NOISE_SHAPE:-paper}"
SHADOW_MODELS="${FEDFED_SHADOW_MODELS:-3}"
SHADOW_TRAIN_SAMPLES="${FEDFED_SHADOW_TRAIN_SAMPLES:-500}"
SHADOW_TEST_SAMPLES="${FEDFED_SHADOW_TEST_SAMPLES:-500}"
SHADOW_EPOCHS="${FEDFED_SHADOW_EPOCHS:-0}"
MIA_SAMPLES="${FEDFED_MIA_SAMPLES:-1000}"
MIA_EPOCHS="${FEDFED_MIA_EPOCHS:-30}"
DIAGNOSTIC_EPOCHS="${FEDFED_DIAGNOSTIC_EPOCHS:-6}"
DIAGNOSTIC_TRAIN_LIMIT="${FEDFED_DIAGNOSTIC_TRAIN_LIMIT:-12000}"
DIAGNOSTIC_TEST_LIMIT="${FEDFED_DIAGNOSTIC_TEST_LIMIT:-4000}"
INVERSION_EPOCHS="${FEDFED_INVERSION_EPOCHS:-8}"
PROBE_LR="${FEDFED_PROBE_LR:-0.001}"
DYNAMIC_NOISE_TRAIN="${FEDFED_DYNAMIC_NOISE_TRAIN:-false}"

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
    --clip-norm)
      CLIP_NORM="$2"
      shift 2
      ;;
    --shadow-models)
      SHADOW_MODELS="$2"
      shift 2
      ;;
    --shadow-train-samples)
      SHADOW_TRAIN_SAMPLES="$2"
      shift 2
      ;;
    --shadow-test-samples)
      SHADOW_TEST_SAMPLES="$2"
      shift 2
      ;;
    --shadow-epochs)
      SHADOW_EPOCHS="$2"
      shift 2
      ;;
    --mia-samples)
      MIA_SAMPLES="$2"
      shift 2
      ;;
    --mia-epochs)
      MIA_EPOCHS="$2"
      shift 2
      ;;
    --diagnostic-epochs)
      DIAGNOSTIC_EPOCHS="$2"
      shift 2
      ;;
    --diagnostic-train-limit)
      DIAGNOSTIC_TRAIN_LIMIT="$2"
      shift 2
      ;;
    --diagnostic-test-limit)
      DIAGNOSTIC_TEST_LIMIT="$2"
      shift 2
      ;;
    --inversion-epochs)
      INVERSION_EPOCHS="$2"
      shift 2
      ;;
    --probe-lr)
      PROBE_LR="$2"
      shift 2
      ;;
    --dynamic-noise-train)
      DYNAMIC_NOISE_TRAIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT

WORKER_FILE="${TMP_DIR}/fedfed_privacy_probe_${RUN_ID}.ps1"
LAUNCHER_FILE="${TMP_DIR}/fedfed_privacy_probe_launcher_${RUN_ID}.ps1"
REMOTE_WORKER_PATH="${REMOTE_Codex_DIR}/fedfed_privacy_probe_${RUN_ID}.ps1"
REMOTE_LAUNCHER_PATH="${REMOTE_Codex_DIR}/fedfed_privacy_probe_launcher_${RUN_ID}.ps1"
REMOTE_WORKER_PS_PATH="${REMOTE_WORKER_PATH//\//\\}"

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
\$env:FEDFED_PRIVACY_VARIANTS = '${VARIANTS}'
\$env:FEDFED_DYNAMIC_NOISE_TRAIN = '${DYNAMIC_NOISE_TRAIN}'

New-Item -ItemType Directory -Force -Path \$runDir | Out-Null
[ordered]@{
  run_id = \$runId
  status = 'running'
  started_utc = (Get-Date).ToUniversalTime().ToString('o')
  variants = \$env:FEDFED_PRIVACY_VARIANTS
  stdout = \$stdoutPath
  stderr = \$stderrPath
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath

try {
  Set-Location \$projectPath
  \$args = @(
    '--dataset_name', 'cifar10',
    '--model_name', 'cifar_resnet18',
    '--round_num', '1',
    '--num_of_clients', '10',
    '--c_fraction', '0.5',
    '--local_epoch', '1',
    '--batch_size', '64',
    '--dataloader_num_workers', '0',
    '--dataloader_pin_memory', 'true',
    '--torch_cudnn_benchmark', 'true',
    '--gpu', 'true',
    '--partition_strategy', 'dirichlet',
    '--dirichlet_alpha', '0.1',
    '--min_samples_per_client', '32',
    '--enable_quantity_skew', 'false',
    '--enable_feature_skew', 'false',
    '--lr', '${PROBE_LR}',
    '--optimizer_name', 'sgd',
    '--weight_decay', '0.0001',
    '--momentum', '0.9',
    '--plugin_name', 'fedfed_image',
    '--fedfed_generator_type', 'paper_beta_vae',
    '--fedfed_vae_latent_channels', '32',
    '--fedfed_vae_z_dim', '2048',
    '--fedfed_distill_rounds', '15',
    '--fedfed_distill_local_epoch', '1',
    '--fedfed_distill_optimizer', 'adamw',
    '--fedfed_distill_lr', '0.001',
    '--fedfed_distill_weight_decay', '0.000001',
    '--fedfed_lambda_recon', '0.2',
    '--fedfed_lambda_fd', '1.5',
    '--fedfed_beta_kl', '0.005',
    '--fedfed_lambda_x_ce', '0.4',
    '--fedfed_lambda_logit_align', '0.5',
    '--fedfed_logit_align_temperature', '2.0',
    '--fedfed_use_augmentation', 'true',
    '--fedfed_mixup_alpha', '2.0',
    '--fedfed_mosaic_batch_size', '64',
    '--fedfed_shared_resident_device', 'cuda',
    '--fedfed_shared_cpu_float16', 'false',
    '--fedfed_clip_norm', '${CLIP_NORM}',
    '--fedfed_noise_shape', '${NOISE_SHAPE}',
    '--privacy_shadow_models', '${SHADOW_MODELS}',
    '--privacy_shadow_train_samples', '${SHADOW_TRAIN_SAMPLES}',
    '--privacy_shadow_test_samples', '${SHADOW_TEST_SAMPLES}',
    '--privacy_shadow_epochs', '${SHADOW_EPOCHS}',
    '--privacy_mia_samples', '${MIA_SAMPLES}',
    '--privacy_mia_epochs', '${MIA_EPOCHS}',
    '--privacy_inversion_epochs', '${INVERSION_EPOCHS}',
    '--diagnostic_epochs', '${DIAGNOSTIC_EPOCHS}',
    '--diagnostic_train_limit', '${DIAGNOSTIC_TRAIN_LIMIT}',
    '--diagnostic_test_limit', '${DIAGNOSTIC_TEST_LIMIT}',
    '--experiment_tag', \$runId
  )
  & \$pythonPath '-u' 'scripts\\run_fedfed_privacy_probe.py' @args 1> \$stdoutPath 2> \$stderrPath
  \$exitCode = \$LASTEXITCODE
} catch {
  Add-Content -Path \$stderrPath -Value (\$_ | Out-String)
  \$exitCode = 1
}

\$finalStatus = if (\$exitCode -eq 0) { 'succeeded' } else { 'failed' }
[ordered]@{
  run_id = \$runId
  status = \$finalStatus
  exit_code = \$exitCode
  finished_utc = (Get-Date).ToUniversalTime().ToString('o')
  result_dir = Join-Path \$projectPath ('result\\cifar10\\fedfed_privacy_probe_' + \$runId)
  stdout = \$stdoutPath
  stderr = \$stderrPath
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath
exit \$exitCode
EOF

cat > "${LAUNCHER_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$workerPath = '${REMOTE_WORKER_PS_PATH}'
\$commandLine = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + \$workerPath + '"'
\$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = \$commandLine }
if (\$result.ReturnValue -ne 0) {
  throw ("WIN32_PROCESS_CREATE_FAILED return_value={0}" -f \$result.ReturnValue)
}
Write-Output ("PRIVACY_PROBE_OK ${RUN_ID} PID={0}" -f \$result.ProcessId)
EOF

ssh_remote "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_Codex_DIR}' | Out-Null; New-Item -ItemType Directory -Force -Path '${REMOTE_RUNS_DIR}' | Out-Null\""
scp_to_remote "${WORKER_FILE}" "${REMOTE_WORKER_PATH}"
scp_to_remote "${LAUNCHER_FILE}" "${REMOTE_LAUNCHER_PATH}"
ssh_remote "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${REMOTE_LAUNCHER_PATH}\""
echo "FedFed privacy probe submitted: ${RUN_ID}"
echo "Check: ./scripts/check_remote_run.sh --run-id ${RUN_ID}"
