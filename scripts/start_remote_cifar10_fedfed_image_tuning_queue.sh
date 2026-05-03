#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

PREFIX="fedfed_tuning_$(date -u +%Y%m%dT%H%M%SZ)"
ROUND_NUM=150
BATCH_SIZE=256
LOCAL_EPOCH=5

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT

QUEUE_FILE="${TMP_DIR}/fedfed_image_tuning_queue.ps1"
LAUNCHER_FILE="${TMP_DIR}/fedfed_image_tuning_queue_launcher.ps1"
REMOTE_QUEUE_PATH="${REMOTE_Codex_DIR}/fedfed_image_tuning_queue_${PREFIX}.ps1"
REMOTE_LAUNCHER_PATH="${REMOTE_Codex_DIR}/fedfed_image_tuning_queue_launcher_${PREFIX}.ps1"
REMOTE_QUEUE_PS_PATH="${REMOTE_QUEUE_PATH//\//\\}"

cat > "${QUEUE_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$pythonPath = '${REMOTE_PYTHON}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$prefix = '${PREFIX}'
\$queueStatusPath = Join-Path \$runsRoot ("\$prefix" + '_queue_status.json')
\$summaryJsonPath = Join-Path \$runsRoot ("\$prefix" + '_tuning_summary.json')
\$summaryCsvPath = Join-Path \$runsRoot ("\$prefix" + '_tuning_summary.csv')
\$plotPath = Join-Path \$runsRoot ("\$prefix" + '_tuning_best_acc_by_stage.png')
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
    plot = \$plotPath
  } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$queueStatusPath
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

function Run-One(\$stage, \$name, \$pluginName, \$cfg) {
  \$runId = "\${prefix}_\${stage}_\${name}"
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
    '--plugin_name', \$pluginName,
    '--experiment_tag', \$runId
  )
  if (\$pluginName -eq 'fedfed_image') {
    \$trainArgs += Config-Args \$cfg
  }

  \$startedUtc = (Get-Date).ToUniversalTime().ToString('o')
  [ordered]@{
    run_id = \$runId
    stage = \$stage
    name = \$name
    plugin_name = \$pluginName
    cfg = \$cfg
    project_path = \$projectPath
    python_path = \$pythonPath
    stdout_log = \$stdoutPath
    stderr_log = \$stderrPath
    launched_utc = \$startedUtc
    mode = 'fedfed_image_tuning_queue'
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

  Write-QueueStatus 'running' \$runId "Running \$stage / \$name"
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
    stage = \$stage
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
    lambda_shared = if (\$cfg) { \$cfg.lambda_shared } else { '' }
    distill_rounds = if (\$cfg) { \$cfg.distill_rounds } else { '' }
    distill_local_epoch = if (\$cfg) { \$cfg.distill_local_epoch } else { '' }
    upload_per_class = if (\$cfg) { \$cfg.upload_per_class } else { '' }
    upload_per_client = if (\$cfg) { \$cfg.upload_per_client } else { '' }
    shared_buffer_size = if (\$cfg) { \$cfg.shared_buffer_size } else { '' }
    shared_per_class_size = if (\$cfg) { \$cfg.shared_per_class_size } else { '' }
    rho = if (\$cfg) { \$cfg.rho } else { '' }
    lambda_rho = if (\$cfg) { \$cfg.lambda_rho } else { '' }
    generator_type = if (\$cfg) { \$cfg.generator_type } else { '' }
    vae_latent_channels = if (\$cfg) { \$cfg.vae_latent_channels } else { '' }
    beta_kl = if (\$cfg) { \$cfg.beta_kl } else { '' }
  }
  [void]\$rows.Add((New-Object PSObject -Property \$row))
  Write-Summary
  return \$row
}

function Run-Stage(\$stage, \$candidates, \$current) {
  \$stageBestRow = \$null
  \$stageBestCfg = \$null
  foreach (\$candidate in \$candidates) {
    \$cfg = Merge-Config \$current \$candidate.delta
    \$row = Run-One \$stage \$candidate.name 'fedfed_image' \$cfg
    if (\$null -eq \$stageBestRow -or [double]\$row.best_acc -gt [double]\$stageBestRow.best_acc) {
      \$stageBestRow = \$row
      \$stageBestCfg = \$cfg
    }
  }
  [void]\$rows.Add((New-Object PSObject -Property ([ordered]@{
    run_id = "\${prefix}_\${stage}_SELECTED"
    stage = \$stage
    name = 'SELECTED'
    plugin_name = 'fedfed_image'
    status = 'selected'
    duration_min = ''
    final_acc = \$stageBestRow.final_acc
    best_acc = \$stageBestRow.best_acc
    best_round = \$stageBestRow.best_round
    final_round = \$stageBestRow.final_round
    final_loss = \$stageBestRow.final_loss
    best_loss = \$stageBestRow.best_loss
    metrics_path = \$stageBestRow.metrics_path
    lambda_shared = \$stageBestCfg.lambda_shared
    distill_rounds = \$stageBestCfg.distill_rounds
    distill_local_epoch = \$stageBestCfg.distill_local_epoch
    upload_per_class = \$stageBestCfg.upload_per_class
    upload_per_client = \$stageBestCfg.upload_per_client
    shared_buffer_size = \$stageBestCfg.shared_buffer_size
    shared_per_class_size = \$stageBestCfg.shared_per_class_size
    rho = \$stageBestCfg.rho
    lambda_rho = \$stageBestCfg.lambda_rho
    generator_type = \$stageBestCfg.generator_type
    vae_latent_channels = \$stageBestCfg.vae_latent_channels
    beta_kl = \$stageBestCfg.beta_kl
  })))
  Write-Summary
  return \$stageBestCfg
}

function New-Candidate(\$name, \$delta) {
  return [ordered]@{ name = \$name; delta = \$delta }
}

function Build-Plots {
  \$plotScriptPath = Join-Path \$runsRoot ("\$prefix" + '_plot_tuning.py')
  @"
import csv
from pathlib import Path
summary = Path(r'\$summaryCsvPath')
plot = Path(r'\$plotPath')
rows = list(csv.DictReader(summary.open('r', encoding='utf-8-sig')))
rows = [r for r in rows if r.get('status') == 'succeeded']
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stages = ['lambda', 'distill', 'pool', 'rho', 'generator']
    fig, axes = plt.subplots(len(stages), 1, figsize=(12, 14), constrained_layout=True)
    for ax, stage in zip(axes, stages):
        subset = [r for r in rows if r.get('stage') == stage]
        labels = [r['name'] for r in subset]
        vals = [100 * float(r['best_acc']) for r in subset]
        colors = ['#4C78A8'] * len(vals)
        ax.bar(labels, vals, color=colors)
        ax.set_title(stage)
        ax.set_ylabel('Best Acc (%)')
        ax.grid(axis='y', alpha=0.25)
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    fig.savefig(plot, dpi=180)
    print('PLOT_OK', plot)
except Exception as exc:
    Path(str(plot) + '.error.txt').write_text(str(exc), encoding='utf-8')
    print('PLOT_FAILED', exc)
"@ | Set-Content -Encoding UTF8 \$plotScriptPath
  & \$pythonPath \$plotScriptPath | Out-File -Encoding UTF8 (Join-Path \$runsRoot ("\$prefix" + '_plot_tuning.log'))
}

try {
  New-Item -ItemType Directory -Force -Path \$runsRoot | Out-Null
  Write-QueueStatus 'starting' '' 'FedFed 21-run staged tuning queue starting.'
  \$current = [ordered]@{
    lambda_shared = '0.2'
    distill_rounds = '30'
    distill_local_epoch = '2'
    upload_per_class = '20'
    upload_per_client = '200'
    shared_buffer_size = '4000'
    shared_per_class_size = '400'
    rho = '0.3'
    lambda_rho = '10'
    generator_type = 'beta_vae'
    vae_latent_channels = '64'
    beta_kl = '0.001'
  }

  [void](Run-One 'baseline' 'fedavg' 'none' \$null)

  \$current = Run-Stage 'lambda' @(
    (New-Candidate 'lambda_shared_0p2' @{ lambda_shared = '0.2' }),
    (New-Candidate 'lambda_shared_0p5' @{ lambda_shared = '0.5' }),
    (New-Candidate 'lambda_shared_1p0' @{ lambda_shared = '1.0' }),
    (New-Candidate 'lambda_shared_2p0' @{ lambda_shared = '2.0' })
  ) \$current

  \$current = Run-Stage 'distill' @(
    (New-Candidate 'distill_30x2' @{ distill_rounds = '30'; distill_local_epoch = '2' }),
    (New-Candidate 'distill_45x2' @{ distill_rounds = '45'; distill_local_epoch = '2' }),
    (New-Candidate 'distill_60x2' @{ distill_rounds = '60'; distill_local_epoch = '2' }),
    (New-Candidate 'distill_30x3' @{ distill_rounds = '30'; distill_local_epoch = '3' })
  ) \$current

  \$current = Run-Stage 'pool' @(
    (New-Candidate 'C3_pool_4000' @{ upload_per_class = '20'; upload_per_client = '200'; shared_buffer_size = '4000'; shared_per_class_size = '400' }),
    (New-Candidate 'C4_pool_8000' @{ upload_per_class = '40'; upload_per_client = '400'; shared_buffer_size = '8000'; shared_per_class_size = '800' }),
    (New-Candidate 'C5_pool_20000' @{ upload_per_class = '100'; upload_per_client = '1000'; shared_buffer_size = '20000'; shared_per_class_size = '2000' })
  ) \$current

  \$current = Run-Stage 'rho' @(
    (New-Candidate 'D1_rho0p2_lrho10' @{ rho = '0.2'; lambda_rho = '10' }),
    (New-Candidate 'D2_rho0p3_lrho10' @{ rho = '0.3'; lambda_rho = '10' }),
    (New-Candidate 'D3_rho0p4_lrho10' @{ rho = '0.4'; lambda_rho = '10' }),
    (New-Candidate 'D4_rho0p3_lrho5' @{ rho = '0.3'; lambda_rho = '5' }),
    (New-Candidate 'D5_rho0p3_lrho20' @{ rho = '0.3'; lambda_rho = '20' })
  ) \$current

  \$current = Run-Stage 'generator' @(
    (New-Candidate 'E1_beta_vae64' @{ generator_type = 'beta_vae'; vae_latent_channels = '64'; beta_kl = '0.001' }),
    (New-Candidate 'E2_beta_vae128' @{ generator_type = 'beta_vae'; vae_latent_channels = '128'; beta_kl = '0.001' }),
    (New-Candidate 'E3_resnet64' @{ generator_type = 'resnet'; vae_latent_channels = '64'; beta_kl = '0.0' }),
    (New-Candidate 'E4_autoencoder64' @{ generator_type = 'autoencoder'; vae_latent_channels = '64'; beta_kl = '0.0' })
  ) \$current

  Build-Plots
  Write-QueueStatus 'succeeded' '' 'FedFed 21-run staged tuning queue finished.'
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
echo "FedFed image tuning queue submitted: ${PREFIX}"
