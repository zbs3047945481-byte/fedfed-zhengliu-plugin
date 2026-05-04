#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

PREFIX="fedavg_fedfed_paper_compare_$(date -u +%Y%m%dT%H%M%SZ)"

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

QUEUE_FILE="${TMP_DIR}/fedavg_fedfed_paper_compare.ps1"
LAUNCHER_FILE="${TMP_DIR}/fedavg_fedfed_paper_compare_launcher.ps1"
REMOTE_QUEUE_PATH="${REMOTE_Codex_DIR}/fedavg_fedfed_paper_compare_${PREFIX}.ps1"
REMOTE_LAUNCHER_PATH="${REMOTE_Codex_DIR}/fedavg_fedfed_paper_compare_launcher_${PREFIX}.ps1"
REMOTE_QUEUE_PS_PATH="${REMOTE_QUEUE_PATH//\//\\}"

cat > "${QUEUE_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$pythonPath = '${REMOTE_PYTHON}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$prefix = '${PREFIX}'
\$queueStatusPath = Join-Path \$runsRoot ("\$prefix" + '_queue_status.json')
\$summaryJsonPath = Join-Path \$runsRoot ("\$prefix" + '_summary.json')
\$summaryCsvPath = Join-Path \$runsRoot ("\$prefix" + '_summary.csv')
\$rows = New-Object System.Collections.ArrayList

function Write-QueueStatus(\$status, \$currentRun, \$message) {
  [ordered]@{
    prefix = \$prefix
    status = \$status
    current_run = \$currentRun
    message = \$message
    updated_utc = (Get-Date).ToUniversalTime().ToString('o')
    summary_json = \$summaryJsonPath
    summary_csv = \$summaryCsvPath
  } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$queueStatusPath
}

function Write-Summary {
  \$rows | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 \$summaryJsonPath
  if (\$rows.Count -gt 0) {
    \$rows | Export-Csv -NoTypeInformation -Encoding UTF8 \$summaryCsvPath
  }
}

function Find-Metrics(\$runId) {
  \$metricsRoot = Join-Path \$projectPath 'result\\cifar10'
  if (-not (Test-Path \$metricsRoot)) { return \$null }
  \$dir = Get-ChildItem -Path \$metricsRoot -Directory -Filter ("*" + \$runId + "*") |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if (\$null -eq \$dir) { return \$null }
  \$metricsPath = Join-Path \$dir.FullName 'metrics.json'
  if (-not (Test-Path \$metricsPath)) { return \$null }
  return \$metricsPath
}

function Parse-Metrics(\$metricsPath) {
  \$m = Get-Content -Raw -LiteralPath \$metricsPath | ConvertFrom-Json
  \$acc = @(\$m.acc_on_g_test_data)
  \$bestRound = 0
  \$best = -1.0
  for (\$i = 0; \$i -lt \$acc.Count; \$i++) {
    if ([double]\$acc[\$i] -gt \$best) {
      \$best = [double]\$acc[\$i]
      \$bestRound = \$i
    }
  }
  return [ordered]@{
    final_acc = [double]\$m.final_test_acc
    best_acc = [double]\$m.best_test_acc
    best_round = \$bestRound
    final_round = [int]\$m.final_round
    final_loss = [double]\$m.final_test_loss
    best_loss = [double]\$m.best_test_loss
  }
}

function Run-One(\$name, \$pluginName) {
  \$runId = "\${prefix}_\${name}"
  \$runDir = Join-Path \$runsRoot \$runId
  \$stdoutPath = Join-Path \$runDir 'stdout.log'
  \$stderrPath = Join-Path \$runDir 'stderr.log'
  \$metaPath = Join-Path \$runDir 'run_meta.json'
  \$statusPath = Join-Path \$runDir 'run_status.json'
  New-Item -ItemType Directory -Force -Path \$runDir | Out-Null
  New-Item -ItemType File -Force -Path \$stdoutPath | Out-Null
  New-Item -ItemType File -Force -Path \$stderrPath | Out-Null

  \$trainArgs = @(
    '--round_num', '300',
    '--num_of_clients', '20',
    '--c_fraction', '0.2',
    '--local_epoch', '5',
    '--batch_size', '64',
    '--dataloader_num_workers', '0',
    '--dataloader_pin_memory', 'true',
    '--torch_cudnn_benchmark', 'true',
    '--gpu', 'true',
    '--dataset_name', 'cifar10',
    '--partition_strategy', 'dirichlet',
    '--dirichlet_alpha', '0.3',
    '--min_samples_per_client', '1',
    '--enable_quantity_skew', 'true',
    '--enable_feature_skew', 'false',
    '--lr', '0.001',
    '--early_stop_enable', 'true',
    '--early_stop_min_rounds', '100',
    '--early_stop_patience', '40',
    '--early_stop_min_delta', '0.001',
    '--plugin_name', \$pluginName,
    '--experiment_tag', \$runId
  )
  if (\$pluginName -eq 'fedfed_image') {
    \$trainArgs += @(
      '--fedfed_two_stage', 'true',
      '--fedfed_generator_type', 'paper_beta_vae',
      '--fedfed_vae_latent_channels', '32',
      '--fedfed_vae_z_dim', '2048',
      '--fedfed_distill_rounds', '15',
      '--fedfed_distill_local_epoch', '1',
      '--fedfed_distill_optimizer', 'adamw',
      '--fedfed_distill_lr', '0.001',
      '--fedfed_distill_weight_decay', '0.000001',
      '--fedfed_lambda_recon', '5.0',
      '--fedfed_lambda_fd', '2.0',
      '--fedfed_beta_kl', '0.005',
      '--fedfed_lambda_x_ce', '0.4',
      '--fedfed_upload_per_class', '0',
      '--fedfed_upload_per_client', '0',
      '--fedfed_shared_buffer_size', '0',
      '--fedfed_shared_per_class_size', '0',
      '--fedfed_shared_batch_size', '0'
    )
  }

  \$startedUtc = (Get-Date).ToUniversalTime().ToString('o')
  [ordered]@{
    run_id = \$runId
    name = \$name
    plugin_name = \$pluginName
    project_path = \$projectPath
    python_path = \$pythonPath
    stdout_log = \$stdoutPath
    stderr_log = \$stderrPath
    launched_utc = \$startedUtc
    arguments = \$trainArgs
  } | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 \$metaPath

  [ordered]@{
    run_id = \$runId
    status = 'running'
    pid = \$PID
    started_utc = \$startedUtc
    exit_code = \$null
    finished_utc = \$null
  } | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 \$statusPath

  Write-QueueStatus 'running' \$runId "Running \$name"
  Set-Location \$projectPath
  \$ErrorActionPreference = 'Continue'
  & \$pythonPath '-u' 'main.py' @trainArgs 1>> \$stdoutPath 2>> \$stderrPath
  \$exitCode = if (\$LASTEXITCODE -ne \$null) { \$LASTEXITCODE } else { 0 }
  \$ErrorActionPreference = 'Stop'
  \$finishedUtc = (Get-Date).ToUniversalTime().ToString('o')
  \$finalStatus = if (\$exitCode -eq 0) { 'succeeded' } else { 'failed' }
  [ordered]@{
    run_id = \$runId
    status = \$finalStatus
    pid = \$PID
    started_utc = \$startedUtc
    exit_code = \$exitCode
    finished_utc = \$finishedUtc
  } | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 \$statusPath
  if (\$exitCode -ne 0) {
    Write-QueueStatus 'failed' \$runId "Training failed with exit code \$exitCode."
    throw "Run failed: \$runId"
  }
  \$metricsPath = Find-Metrics \$runId
  if (\$null -eq \$metricsPath) {
    Write-QueueStatus 'failed' \$runId 'metrics.json not found after successful process exit.'
    throw "metrics.json not found: \$runId"
  }
  \$metrics = Parse-Metrics \$metricsPath
  \$durationMin = [Math]::Round(((Get-Date \$finishedUtc) - (Get-Date \$startedUtc)).TotalMinutes, 2)
  \$row = [ordered]@{
    run_id = \$runId
    name = \$name
    plugin_name = \$pluginName
    status = \$finalStatus
    duration_min = \$durationMin
    final_acc = \$metrics.final_acc
    best_acc = \$metrics.best_acc
    best_round = \$metrics.best_round
    final_round = \$metrics.final_round
    final_loss = \$metrics.final_loss
    best_loss = \$metrics.best_loss
    metrics_path = \$metricsPath
  }
  [void]\$rows.Add((New-Object PSObject -Property \$row))
  Write-Summary
}

try {
  New-Item -ItemType Directory -Force -Path \$runsRoot | Out-Null
  Write-QueueStatus 'starting' '' 'FedAvg vs paper-style FedFed comparison starting.'
  Run-One 'fedavg' 'none'
  Run-One 'fedfed_paper' 'fedfed_image'
  Write-QueueStatus 'succeeded' '' 'FedAvg vs paper-style FedFed comparison finished.'
} catch {
  Add-Content -Path (Join-Path \$runsRoot ("\$prefix" + '_queue_error.log')) -Value (\$_ | Out-String)
  Write-Summary
  Write-QueueStatus 'failed' '' (\$_ | Out-String)
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
echo "FedAvg vs paper FedFed comparison submitted: ${PREFIX}"
