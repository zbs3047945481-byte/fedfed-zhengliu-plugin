#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

PREFIX="fedfed_xs_diag_$(date -u +%Y%m%dT%H%M%SZ)"

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

RUN_FILE="${TMP_DIR}/fedfed_xs_diagnostic_${PREFIX}.ps1"
LAUNCHER_FILE="${TMP_DIR}/fedfed_xs_diagnostic_launcher_${PREFIX}.ps1"
REMOTE_RUN_PATH="${REMOTE_Codex_DIR}/fedfed_xs_diagnostic_${PREFIX}.ps1"
REMOTE_LAUNCHER_PATH="${REMOTE_Codex_DIR}/fedfed_xs_diagnostic_launcher_${PREFIX}.ps1"
REMOTE_RUN_PS_PATH="${REMOTE_RUN_PATH//\//\\}"

cat > "${RUN_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$pythonPath = '${REMOTE_PYTHON}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$prefix = '${PREFIX}'
\$runDir = Join-Path \$runsRoot \$prefix
\$stdoutPath = Join-Path \$runDir 'stdout.log'
\$stderrPath = Join-Path \$runDir 'stderr.log'
\$statusPath = Join-Path \$runsRoot ("\$prefix" + '_queue_status.json')
New-Item -ItemType Directory -Force -Path \$runDir | Out-Null

function Write-Status(\$status, \$message) {
  [ordered]@{
    prefix = \$prefix
    status = \$status
    message = \$message
    updated_utc = (Get-Date).ToUniversalTime().ToString('o')
    run_dir = \$runDir
    stdout = \$stdoutPath
    stderr = \$stderrPath
    result_dir = Join-Path \$projectPath ('result\\cifar10\\fedfed_xs_diagnostic_' + \$prefix)
  } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath
}

try {
  Write-Status 'running' 'FedFed x/x_s/x_r diagnostic running.'
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
    '--fedfed_lambda_recon', '5.0',
    '--fedfed_lambda_fd', '2.0',
    '--fedfed_beta_kl', '0.005',
    '--fedfed_lambda_x_ce', '0.4',
    '--fedfed_distill_rounds', '15',
    '--fedfed_distill_local_epoch', '1',
    '--fedfed_rho', '0.4',
    '--fedfed_lambda_rho', '10',
    '--diagnostic_epochs', '10',
    '--diagnostic_train_limit', '20000',
    '--diagnostic_test_limit', '10000',
    '--experiment_tag', \$prefix
  )
  & \$pythonPath '-u' 'scripts\\run_fedfed_xs_diagnostic.py' @args 1> \$stdoutPath 2> \$stderrPath
  if (\$LASTEXITCODE -ne 0) {
    throw "Diagnostic failed with exit code \$LASTEXITCODE"
  }
  Write-Status 'succeeded' 'FedFed x/x_s/x_r diagnostic finished.'
} catch {
  Add-Content -Path (Join-Path \$runDir 'queue_error.log') -Value (\$_ | Out-String)
  Write-Status 'failed' (\$_ | Out-String)
  exit 1
}
EOF

cat > "${LAUNCHER_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$queuePath = '${REMOTE_RUN_PS_PATH}'
\$commandLine = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + \$queuePath + '"'
\$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = \$commandLine }
if (\$result.ReturnValue -ne 0) {
  throw ("WIN32_PROCESS_CREATE_FAILED return_value={0}" -f \$result.ReturnValue)
}
Write-Output ("QUEUE_OK ${PREFIX} PID={0}" -f \$result.ProcessId)
EOF

ssh_remote "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_Codex_DIR}' | Out-Null\""
scp_to_remote "${RUN_FILE}" "${REMOTE_RUN_PATH}"
scp_to_remote "${LAUNCHER_FILE}" "${REMOTE_LAUNCHER_PATH}"
ssh_remote "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${REMOTE_LAUNCHER_PATH}\""
echo "FedFed x/x_s/x_r diagnostic submitted: ${PREFIX}"
