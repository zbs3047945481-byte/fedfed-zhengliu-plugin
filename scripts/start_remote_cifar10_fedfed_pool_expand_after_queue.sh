#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

WAIT_PREFIX="fedfed_plugin_tuning_quantity_round150_20260502T105738Z"
PREFIX="fedfed_pool_expand_quantity_round150_$(date -u +%Y%m%dT%H%M%SZ)"
ROUND_NUM=150
BATCH_SIZE=256
LOCAL_EPOCH=5
POLL_SECONDS=60

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait-prefix)
      WAIT_PREFIX="$2"
      shift 2
      ;;
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --round-num)
      ROUND_NUM="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --local-epoch)
      LOCAL_EPOCH="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
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

QUEUE_FILE="${TMP_DIR}/fedfed_pool_expand_after_queue.ps1"
LAUNCHER_FILE="${TMP_DIR}/fedfed_pool_expand_after_queue_launcher.ps1"
REMOTE_QUEUE_PATH="${REMOTE_Codex_DIR}/fedfed_pool_expand_after_queue_${PREFIX}.ps1"
REMOTE_LAUNCHER_PATH="${REMOTE_Codex_DIR}/fedfed_pool_expand_after_queue_launcher_${PREFIX}.ps1"
REMOTE_QUEUE_PS_PATH="${REMOTE_QUEUE_PATH//\//\\}"

cat > "${QUEUE_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$pythonPath = '${REMOTE_PYTHON}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$waitPrefix = '${WAIT_PREFIX}'
\$prefix = '${PREFIX}'
\$pollSeconds = ${POLL_SECONDS}
\$waitStatusPath = Join-Path \$runsRoot ("\$waitPrefix" + '_queue_status.json')
\$queueStatusPath = Join-Path \$runsRoot ("\$prefix" + '_queue_status.json')
\$summaryJsonPath = Join-Path \$runsRoot ("\$prefix" + '_pool_expand_summary.json')
\$summaryCsvPath = Join-Path \$runsRoot ("\$prefix" + '_pool_expand_summary.csv')
\$plotPath = Join-Path \$runsRoot ("\$prefix" + '_pool_expand_best_acc.png')
\$rows = New-Object System.Collections.ArrayList

function Write-QueueStatus(\$status, \$currentRun, \$message) {
  [ordered]@{
    prefix = \$prefix
    status = \$status
    current_run = \$currentRun
    message = \$message
    wait_prefix = \$waitPrefix
    wait_status_path = \$waitStatusPath
    updated_utc = (Get-Date).ToUniversalTime().ToString('o')
    summary_json = \$summaryJsonPath
    summary_csv = \$summaryCsvPath
    plot = \$plotPath
  } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$queueStatusPath
}

function Wait-For-PreviousQueue {
  while (\$true) {
    if (-not (Test-Path \$waitStatusPath)) {
      Write-QueueStatus 'waiting' '' "Waiting for status file: \$waitStatusPath"
      Start-Sleep -Seconds \$pollSeconds
      continue
    }
    \$statusObj = Get-Content -Raw -LiteralPath \$waitStatusPath | ConvertFrom-Json
    \$status = [string]\$statusObj.status
    \$current = [string]\$statusObj.current_run
    if (@('running', 'starting', 'waiting') -contains \$status) {
      Write-QueueStatus 'waiting' \$current "Previous queue still \$status; polling every \$pollSeconds seconds."
      Start-Sleep -Seconds \$pollSeconds
      continue
    }
    Write-QueueStatus 'starting' '' "Previous queue ended with status '\$status'; starting pool expansion queue."
    return \$status
  }
}

function Copy-Config(\$cfg) {
  \$out = @{}
  foreach (\$key in \$cfg.Keys) { \$out[\$key] = [string]\$cfg[\$key] }
  return \$out
}

function Merge-Config(\$base, \$delta) {
  \$out = Copy-Config \$base
  foreach (\$key in \$delta.Keys) { \$out[\$key] = [string]\$delta[\$key] }
  return \$out
}

function Config-Args(\$cfg) {
  return @(
    '--fedfed_lambda_shared', \$cfg.lambda_shared,
    '--fedfed_distill_rounds', \$cfg.distill_rounds,
    '--fedfed_distill_local_epoch', \$cfg.distill_local_epoch,
    '--fedfed_upload_per_class', \$cfg.upload_per_class,
    '--fedfed_upload_per_client', \$cfg.upload_per_client,
    '--fedfed_shared_buffer_size', \$cfg.shared_buffer_size,
    '--fedfed_shared_per_class_size', \$cfg.shared_per_class_size,
    '--fedfed_rho', \$cfg.rho,
    '--fedfed_lambda_rho', \$cfg.lambda_rho,
    '--fedfed_generator_type', \$cfg.generator_type,
    '--fedfed_vae_latent_channels', \$cfg.vae_latent_channels,
    '--fedfed_beta_kl', \$cfg.beta_kl
  )
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

function Run-One(\$name, \$cfg) {
  \$runId = "\${prefix}_pool_\${name}"
  \$runDir = Join-Path \$runsRoot \$runId
  \$stdoutPath = Join-Path \$runDir 'stdout.log'
  \$stderrPath = Join-Path \$runDir 'stderr.log'
  \$metaPath = Join-Path \$runDir 'run_meta.json'
  \$statusPath = Join-Path \$runDir 'run_status.json'
  New-Item -ItemType Directory -Force -Path \$runDir | Out-Null
  New-Item -ItemType File -Force -Path \$stdoutPath | Out-Null
  New-Item -ItemType File -Force -Path \$stderrPath | Out-Null

  \$trainArgs = @(
    '--round_num', '${ROUND_NUM}',
    '--num_of_clients', '20',
    '--c_fraction', '0.2',
    '--local_epoch', '${LOCAL_EPOCH}',
    '--batch_size', '${BATCH_SIZE}',
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
    '--early_stop_min_rounds', '40',
    '--early_stop_patience', '20',
    '--early_stop_min_delta', '0.002',
    '--plugin_name', 'fedfed_image',
    '--experiment_tag', \$runId
  )
  \$trainArgs += Config-Args \$cfg

  \$startedUtc = (Get-Date).ToUniversalTime().ToString('o')
  [ordered]@{
    run_id = \$runId
    name = \$name
    plugin_name = 'fedfed_image'
    cfg = \$cfg
    project_path = \$projectPath
    python_path = \$pythonPath
    stdout_log = \$stdoutPath
    stderr_log = \$stderrPath
    launched_utc = \$startedUtc
    mode = 'fedfed_pool_expand_after_queue'
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

  Write-QueueStatus 'running' \$runId "Running pool expansion \$name"
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
    status = \$finalStatus
    duration_min = \$durationMin
    final_acc = \$metrics.final_acc
    best_acc = \$metrics.best_acc
    best_round = \$metrics.best_round
    final_round = \$metrics.final_round
    final_loss = \$metrics.final_loss
    best_loss = \$metrics.best_loss
    metrics_path = \$metricsPath
    lambda_shared = \$cfg.lambda_shared
    distill_rounds = \$cfg.distill_rounds
    distill_local_epoch = \$cfg.distill_local_epoch
    upload_per_class = \$cfg.upload_per_class
    upload_per_client = \$cfg.upload_per_client
    shared_buffer_size = \$cfg.shared_buffer_size
    shared_per_class_size = \$cfg.shared_per_class_size
    rho = \$cfg.rho
    lambda_rho = \$cfg.lambda_rho
    generator_type = \$cfg.generator_type
    vae_latent_channels = \$cfg.vae_latent_channels
    beta_kl = \$cfg.beta_kl
  }
  [void]\$rows.Add((New-Object PSObject -Property \$row))
  Write-Summary
}

function New-Candidate(\$name, \$delta) {
  return [ordered]@{ name = \$name; delta = \$delta }
}

function Build-Plot {
  \$plotScriptPath = Join-Path \$runsRoot ("\$prefix" + '_plot_pool_expand.py')
  @"
import csv
from pathlib import Path
summary = Path(r'\$summaryCsvPath')
plot = Path(r'\$plotPath')
rows = [r for r in csv.DictReader(summary.open('r', encoding='utf-8-sig')) if r.get('status') == 'succeeded']
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [r['name'] for r in rows]
    best = [100 * float(r['best_acc']) for r in rows]
    final = [100 * float(r['final_acc']) for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(x, best, marker='o', label='Best Acc')
    ax.plot(x, final, marker='s', label='Final Acc')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('FedFed Pool Expansion')
    ax.grid(alpha=0.25)
    ax.legend()
    for i, v in enumerate(best):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    fig.savefig(plot, dpi=180)
    print('PLOT_OK', plot)
except Exception as exc:
    Path(str(plot) + '.error.txt').write_text(str(exc), encoding='utf-8')
    print('PLOT_FAILED', exc)
"@ | Set-Content -Encoding UTF8 \$plotScriptPath
  & \$pythonPath \$plotScriptPath | Out-File -Encoding UTF8 (Join-Path \$runsRoot ("\$prefix" + '_plot_pool_expand.log'))
}

try {
  New-Item -ItemType Directory -Force -Path \$runsRoot | Out-Null
  \$previousStatus = Wait-For-PreviousQueue
  \$base = [ordered]@{
    lambda_shared = '1.0'
    distill_rounds = '30'
    distill_local_epoch = '2'
    upload_per_class = '100'
    upload_per_client = '1000'
    shared_buffer_size = '20000'
    shared_per_class_size = '2000'
    rho = '0.4'
    lambda_rho = '10'
    generator_type = 'beta_vae'
    vae_latent_channels = '64'
    beta_kl = '0.001'
  }

  \$candidates = @(
    (New-Candidate 'C5_up100_buf20000' @{ upload_per_class = '100'; upload_per_client = '1000'; shared_buffer_size = '20000'; shared_per_class_size = '2000' }),
    (New-Candidate 'C6_up150_buf30000' @{ upload_per_class = '150'; upload_per_client = '1500'; shared_buffer_size = '30000'; shared_per_class_size = '3000' }),
    (New-Candidate 'C7_up200_buf40000' @{ upload_per_class = '200'; upload_per_client = '2000'; shared_buffer_size = '40000'; shared_per_class_size = '4000' }),
    (New-Candidate 'C8_up300_buf60000' @{ upload_per_class = '300'; upload_per_client = '3000'; shared_buffer_size = '60000'; shared_per_class_size = '6000' })
  )

  foreach (\$candidate in \$candidates) {
    \$cfg = Merge-Config \$base \$candidate.delta
    Run-One \$candidate.name \$cfg
  }
  Build-Plot
  Write-QueueStatus 'succeeded' '' "Pool expansion queue finished. Previous queue status was '\$previousStatus'."
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
echo "FedFed pool expansion queue submitted: ${PREFIX}"
echo "Waiting for previous queue: ${WAIT_PREFIX}"
