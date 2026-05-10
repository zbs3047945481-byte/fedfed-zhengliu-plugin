#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

PREFIX="fedfed_recon_noisy_diag_$(date -u +%Y%m%dT%H%M%SZ)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
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

QUEUE_FILE="${TMP_DIR}/fedfed_recon_noisy_diag_${PREFIX}.ps1"
LAUNCHER_FILE="${TMP_DIR}/fedfed_recon_noisy_diag_launcher_${PREFIX}.ps1"
REMOTE_QUEUE_PATH="${REMOTE_Codex_DIR}/fedfed_recon_noisy_diag_${PREFIX}.ps1"
REMOTE_LAUNCHER_PATH="${REMOTE_Codex_DIR}/fedfed_recon_noisy_diag_launcher_${PREFIX}.ps1"
REMOTE_QUEUE_PS_PATH="${REMOTE_QUEUE_PATH//\//\\}"

cat > "${QUEUE_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$pythonPath = '${REMOTE_PYTHON}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$prefix = '${PREFIX}'
\$queueStatusPath = Join-Path \$runsRoot ("\$prefix" + '_queue_status.json')

function Write-QueueStatus(\$status, \$message, \$currentRun) {
  [ordered]@{
    prefix = \$prefix
    status = \$status
    message = \$message
    current_run = \$currentRun
    updated_utc = (Get-Date).ToUniversalTime().ToString('o')
    runs_root = \$runsRoot
  } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$queueStatusPath
}

function Run-Diagnostic(\$runSuffix, \$lambdaRecon, \$noiseType, \$noiseStd1, \$noiseStd2, \$ceOnNoisyXs) {
  \$runId = "\$prefix" + '_' + \$runSuffix
  \$runDir = Join-Path \$runsRoot \$runId
  \$stdoutPath = Join-Path \$runDir 'stdout.log'
  \$stderrPath = Join-Path \$runDir 'stderr.log'
  \$statusPath = Join-Path \$runDir 'run_status.json'
  New-Item -ItemType Directory -Force -Path \$runDir | Out-Null

  [ordered]@{
    run_id = \$runId
    status = 'running'
    started_utc = (Get-Date).ToUniversalTime().ToString('o')
    lambda_recon = \$lambdaRecon
    noise_type = \$noiseType
    noise_std1 = \$noiseStd1
    noise_std2 = \$noiseStd2
    distill_ce_on_noisy_xs = \$ceOnNoisyXs
    result_dir = Join-Path \$projectPath ('result\\cifar10\\fedfed_xs_diagnostic_' + \$runId)
  } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath

  Write-QueueStatus 'running' ("Running " + \$runId) \$runId
  Set-Location \$projectPath
  \$args = @(
    '--dataset_name', 'cifar10',
    '--round_num', '1',
    '--num_of_clients', '20',
    '--c_fraction', '0.2',
    '--local_epoch', '5',
    '--batch_size', '64',
    '--dataloader_num_workers', '0',
    '--dataloader_pin_memory', 'true',
    '--torch_cudnn_benchmark', 'true',
    '--gpu', 'true',
    '--partition_strategy', 'dirichlet',
    '--dirichlet_alpha', '0.3',
    '--min_samples_per_client', '1',
    '--enable_quantity_skew', 'true',
    '--enable_feature_skew', 'false',
    '--lr', '0.001',
    '--plugin_name', 'fedfed_image',
    '--fedfed_generator_type', 'paper_beta_vae',
    '--fedfed_vae_latent_channels', '32',
    '--fedfed_vae_z_dim', '2048',
    '--fedfed_distill_optimizer', 'adamw',
    '--fedfed_distill_lr', '0.001',
    '--fedfed_distill_weight_decay', '0.000001',
    '--fedfed_lambda_recon', ([string]\$lambdaRecon),
    '--fedfed_lambda_fd', '2.0',
    '--fedfed_beta_kl', '0.005',
    '--fedfed_lambda_x_ce', '0.4',
    '--fedfed_use_augmentation', 'true',
    '--fedfed_mixup_alpha', '2.0',
    '--fedfed_mosaic_batch_size', '64',
    '--fedfed_distill_rounds', '15',
    '--fedfed_distill_local_epoch', '1',
    '--fedfed_noise_type', \$noiseType,
    '--fedfed_noise_mean', '0.0',
    '--fedfed_noise_std1', ([string]\$noiseStd1),
    '--fedfed_noise_std2', ([string]\$noiseStd2),
    '--fedfed_distill_ce_on_noisy_xs', ([string]\$ceOnNoisyXs).ToLower(),
    '--diagnostic_epochs', '10',
    '--diagnostic_train_limit', '20000',
    '--diagnostic_test_limit', '10000',
    '--experiment_tag', \$runId
  )
  \$process = Start-Process -FilePath \$pythonPath -ArgumentList (@('-u', 'scripts\\run_fedfed_xs_diagnostic.py') + \$args) -WorkingDirectory \$projectPath -RedirectStandardOutput \$stdoutPath -RedirectStandardError \$stderrPath -NoNewWindow -PassThru -Wait
  if (\$process.ExitCode -ne 0) {
    [ordered]@{
      run_id = \$runId
      status = 'failed'
      exit_code = \$process.ExitCode
      updated_utc = (Get-Date).ToUniversalTime().ToString('o')
      stdout = \$stdoutPath
      stderr = \$stderrPath
    } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath
    throw "Diagnostic \$runId failed with exit code \$(\$process.ExitCode)"
  }
  [ordered]@{
    run_id = \$runId
    status = 'succeeded'
    finished_utc = (Get-Date).ToUniversalTime().ToString('o')
    stdout = \$stdoutPath
    stderr = \$stderrPath
    result_dir = Join-Path \$projectPath ('result\\cifar10\\fedfed_xs_diagnostic_' + \$runId)
  } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath
}

try {
  Write-QueueStatus 'running' 'FedFed recon/noisy diagnostic queue started.' ''
  Run-Diagnostic 'recon0p5_no_noise' 0.5 'none' 0.0 0.0 \$false
  Run-Diagnostic 'recon0p2_no_noise' 0.2 'none' 0.0 0.0 \$false
  Run-Diagnostic 'recon1p0_noisy_ce' 1.0 'gaussian' 0.2 0.25 \$true
  Write-QueueStatus 'succeeded' 'FedFed recon/noisy diagnostic queue finished.' ''
} catch {
  Add-Content -Path (Join-Path \$runsRoot ("\$prefix" + '_queue_error.log')) -Value (\$_ | Out-String)
  Write-QueueStatus 'failed' (\$_ | Out-String) ''
  exit 1
}
EOF

cat > "${LAUNCHER_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$queuePath = '${REMOTE_QUEUE_PS_PATH}'
\$commandLine = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + \$queuePath + '"'
\$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = \$commandLine }
if (\$result.ReturnValue -ne 0) {
  throw ("WIN32_PROCESS_CREATE_FAILED return_value={0}" -f \$result.ReturnValue)
}
Write-Output ("QUEUE_OK ${PREFIX} PID={0}" -f \$result.ProcessId)
EOF

ssh_remote "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_Codex_DIR}' | Out-Null\""
scp_to_remote "${QUEUE_FILE}" "${REMOTE_QUEUE_PATH}"
scp_to_remote "${LAUNCHER_FILE}" "${REMOTE_LAUNCHER_PATH}"
ssh_remote "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${REMOTE_LAUNCHER_PATH}\""
echo "FedFed recon/noisy diagnostic queue submitted: ${PREFIX}"
