#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

PREFIX="fedfed_final_diag_formal_$(date -u +%Y%m%dT%H%M%SZ)"
WAIT_RUN_ID="${WAIT_RUN_ID:-fedfed_align_xce_diag_20260505T120933Z_fd1p5_align0p2_xce0p8}"

TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT

QUEUE_FILE="${TMP_DIR}/fedfed_final_queue_${PREFIX}.ps1"
LAUNCHER_FILE="${TMP_DIR}/fedfed_final_queue_launcher_${PREFIX}.ps1"
PLOT_FILE="${TMP_DIR}/fedfed_final_plot_${PREFIX}.py"
REMOTE_QUEUE_PATH="${REMOTE_Codex_DIR}/fedfed_final_queue_${PREFIX}.ps1"
REMOTE_LAUNCHER_PATH="${REMOTE_Codex_DIR}/fedfed_final_queue_launcher_${PREFIX}.ps1"
REMOTE_PLOT_PATH="${REMOTE_Codex_DIR}/fedfed_final_plot_${PREFIX}.py"
REMOTE_QUEUE_PS_PATH="${REMOTE_QUEUE_PATH//\//\\}"

cat > "${PLOT_FILE}" <<'PY'
import csv
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TEMP", "."), "matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

summary_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
with open(summary_path, "r", encoding="utf-8-sig") as f:
    rows = json.load(f)

with open(os.path.join(out_dir, "formal_summary.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["run_id", "method", "alpha", "local_epoch", "best_acc", "final_acc", "best_round", "final_round", "duration_min", "metrics_path"])
    writer.writeheader()
    for row in rows:
        if row.get("kind") == "formal":
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})

formal = [row for row in rows if row.get("kind") == "formal"]
for epoch in sorted({int(row["local_epoch"]) for row in formal}):
    subset = [row for row in formal if int(row["local_epoch"]) == epoch]
    curves = []
    for row in subset:
        with open(row["metrics_path"], "r", encoding="utf-8") as f:
            metrics = json.load(f)
        curves.append((row["method"], metrics.get("rounds", []), metrics.get("acc_on_g_test_data", [])))
    if curves:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for method, rounds, acc in curves:
            ax.plot(rounds, [v * 100 for v in acc], linewidth=2.0, label=method)
        ax.set_title(f"FedAvg vs FedFed, alpha=0.1, local_epoch={epoch}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Test accuracy (%)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"accuracy_curve_epoch{epoch}.png"), dpi=300)
        fig.savefig(os.path.join(out_dir, f"accuracy_curve_epoch{epoch}.pdf"))
        plt.close(fig)

    labels = [row["method"] for row in subset]
    best = [row["best_acc"] * 100 for row in subset]
    final = [row["final_acc"] * 100 for row in subset]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar([i - 0.18 for i in x], best, width=0.36, label="Best")
    ax.bar([i + 0.18 for i in x], final, width=0.36, label="Final")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(f"Best/Final accuracy, local_epoch={epoch}")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"best_final_epoch{epoch}.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, f"best_final_epoch{epoch}.pdf"))
    plt.close(fig)
PY

cat > "${QUEUE_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$pythonPath = '${REMOTE_PYTHON}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$prefix = '${PREFIX}'
\$waitRunId = '${WAIT_RUN_ID}'
\$plotScript = '${REMOTE_PLOT_PATH}'
\$queueStatusPath = Join-Path \$runsRoot ("\$prefix" + '_queue_status.json')
\$summaryJsonPath = Join-Path \$runsRoot ("\$prefix" + '_summary.json')
\$summaryCsvPath = Join-Path \$runsRoot ("\$prefix" + '_summary.csv')
\$figuresDir = Join-Path \$runsRoot ("\$prefix" + '_figures')
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
    figures_dir = \$figuresDir
  } | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 \$queueStatusPath
}

function Write-Summary {
  \$rows | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 \$summaryJsonPath
  if (\$rows.Count -gt 0) {
    \$rows | Export-Csv -NoTypeInformation -Encoding UTF8 \$summaryCsvPath
  }
}

function Wait-PreviousRun {
  if ([string]::IsNullOrWhiteSpace(\$waitRunId)) { return }
  \$statusPath = Join-Path (Join-Path \$runsRoot \$waitRunId) 'run_status.json'
  while (\$true) {
    if (-not (Test-Path \$statusPath)) {
      Write-QueueStatus 'waiting' '' ('Waiting for status file: ' + \$waitRunId)
      Start-Sleep -Seconds 60
      continue
    }
    \$statusObj = Get-Content \$statusPath -Raw | ConvertFrom-Json
    if (\$statusObj.status -ne 'running') { return }
    Write-QueueStatus 'waiting' '' ('Waiting for running experiment: ' + \$waitRunId)
    Start-Sleep -Seconds 60
  }
}

function Find-Metrics(\$runId) {
  \$metricsRoot = Join-Path \$projectPath 'result\\cifar10'
  if (-not (Test-Path \$metricsRoot)) { return \$null }
  \$dir = Get-ChildItem -Path \$metricsRoot -Directory -Filter ("*" + \$runId + "*") | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (\$null -eq \$dir) { return \$null }
  \$metricsPath = Join-Path \$dir.FullName 'metrics.json'
  if (-not (Test-Path \$metricsPath)) { return \$null }
  return \$metricsPath
}

function Parse-Metrics(\$metricsPath) {
  \$m = Get-Content -Raw -LiteralPath \$metricsPath | ConvertFrom-Json
  \$acc = @(\$m.acc_on_g_test_data)
  \$rounds = @(\$m.rounds)
  \$bestRound = 0
  \$best = -1.0
  for (\$i = 0; \$i -lt \$acc.Count; \$i++) {
    if ([double]\$acc[\$i] -gt \$best) {
      \$best = [double]\$acc[\$i]
      \$bestRound = if (\$rounds.Count -gt \$i) { [int]\$rounds[\$i] } else { \$i }
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

function Parse-Diagnostic(\$runId) {
  \$diagRoot = Join-Path \$projectPath 'result\\cifar10'
  \$dir = Get-ChildItem -Path \$diagRoot -Directory -Filter ("fedfed_xs_diagnostic_" + \$runId) | Select-Object -First 1
  if (\$null -eq \$dir) { throw "diagnostic dir not found: \$runId" }
  \$metricsPath = Join-Path \$dir.FullName 'diagnostic_metrics.json'
  if (-not (Test-Path \$metricsPath)) { throw "diagnostic metrics not found: \$runId" }
  return \$metricsPath
}

function Run-Diagnostic(\$name, \$lambdaFd, \$lambdaAlign, \$lambdaLogitAlign, \$temperature) {
  \$runId = "\$prefix" + '_' + \$name
  \$runDir = Join-Path \$runsRoot \$runId
  \$stdoutPath = Join-Path \$runDir 'stdout.log'
  \$stderrPath = Join-Path \$runDir 'stderr.log'
  \$statusPath = Join-Path \$runDir 'run_status.json'
  New-Item -ItemType Directory -Force -Path \$runDir | Out-Null
  [ordered]@{run_id=\$runId; status='running'; kind='diagnostic'; started_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath
  Write-QueueStatus 'running' \$runId ("Running diagnostic " + \$name)
  Set-Location \$projectPath
  \$args = @(
    '--dataset_name','cifar10','--round_num','1','--num_of_clients','20','--c_fraction','0.2','--local_epoch','5','--batch_size','64',
    '--dataloader_num_workers','0','--dataloader_pin_memory','true','--torch_cudnn_benchmark','true','--gpu','true',
    '--partition_strategy','dirichlet','--dirichlet_alpha','0.3','--min_samples_per_client','1','--enable_quantity_skew','true','--enable_feature_skew','false',
    '--lr','0.001','--plugin_name','fedfed_image','--fedfed_generator_type','paper_beta_vae','--fedfed_vae_latent_channels','32','--fedfed_vae_z_dim','2048',
    '--fedfed_distill_optimizer','adamw','--fedfed_distill_lr','0.001','--fedfed_distill_weight_decay','0.000001',
    '--fedfed_lambda_recon','0.2','--fedfed_lambda_fd',([string]\$lambdaFd),'--fedfed_lambda_align',([string]\$lambdaAlign),
    '--fedfed_lambda_logit_align',([string]\$lambdaLogitAlign),'--fedfed_logit_align_temperature',([string]\$temperature),
    '--fedfed_beta_kl','0.005','--fedfed_lambda_x_ce','0.4','--fedfed_use_augmentation','true','--fedfed_mixup_alpha','2.0','--fedfed_mosaic_batch_size','64',
    '--fedfed_distill_rounds','15','--fedfed_distill_local_epoch','1','--fedfed_noise_type','none','--fedfed_noise_std1','0.0','--fedfed_noise_std2','0.0',
    '--fedfed_distill_ce_on_noisy_xs','false','--diagnostic_epochs','10','--diagnostic_train_limit','20000','--diagnostic_test_limit','10000','--experiment_tag',\$runId
  )
  & \$pythonPath '-u' 'scripts\\run_fedfed_xs_diagnostic.py' @args 1> \$stdoutPath 2> \$stderrPath
  if (\$LASTEXITCODE -ne 0) { throw "diagnostic failed: \$runId" }
  \$metricsPath = Parse-Diagnostic \$runId
  \$j = Get-Content \$metricsPath -Raw | ConvertFrom-Json
  \$row = [ordered]@{kind='diagnostic';run_id=\$runId;name=\$name;metrics_path=\$metricsPath;lambda_fd=\$lambdaFd;lambda_align=\$lambdaAlign;lambda_logit_align=\$lambdaLogitAlign;temperature=\$temperature}
  foreach (\$r in \$j.results) {
    \$row[([string]\$r.mode + '_best')] = [double]\$r.best_acc
    \$row[([string]\$r.mode + '_final')] = [double]\$r.final_acc
  }
  [void]\$rows.Add((New-Object PSObject -Property \$row))
  Write-Summary
  [ordered]@{run_id=\$runId; status='succeeded'; kind='diagnostic'; finished_utc=(Get-Date).ToUniversalTime().ToString('o'); metrics_path=\$metricsPath} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath
}

function Run-Formal(\$method, \$pluginName, \$alpha, \$localEpoch) {
  \$runId = "\$prefix" + '_' + \$method + '_a0p1_ep' + \$localEpoch
  \$runDir = Join-Path \$runsRoot \$runId
  \$stdoutPath = Join-Path \$runDir 'stdout.log'
  \$stderrPath = Join-Path \$runDir 'stderr.log'
  \$statusPath = Join-Path \$runDir 'run_status.json'
  New-Item -ItemType Directory -Force -Path \$runDir | Out-Null
  [ordered]@{run_id=\$runId; status='running'; kind='formal'; started_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath
  Write-QueueStatus 'running' \$runId ("Running formal " + \$method + " epoch=" + \$localEpoch)
  Set-Location \$projectPath
  \$args = @(
    '--round_num','300','--num_of_clients','10','--c_fraction','0.5','--local_epoch',([string]\$localEpoch),'--batch_size','32',
    '--dataloader_num_workers','0','--dataloader_pin_memory','true','--torch_cudnn_benchmark','true','--gpu','true',
    '--dataset_name','cifar10','--partition_strategy','dirichlet','--dirichlet_alpha',([string]\$alpha),'--min_samples_per_client','1',
    '--enable_quantity_skew','true','--enable_feature_skew','false','--lr','0.01','--optimizer_name','sgd','--weight_decay','0.0001',
    '--momentum','0.9','--nesterov','false','--lr_schedule','none','--early_stop_enable','true','--early_stop_min_rounds','150',
    '--early_stop_patience','80','--early_stop_min_delta','0.001','--plugin_name',\$pluginName,'--experiment_tag',\$runId
  )
  if (\$pluginName -eq 'fedfed_image') {
    \$args += @(
      '--fedfed_two_stage','true','--fedfed_generator_type','paper_beta_vae','--fedfed_vae_latent_channels','32','--fedfed_vae_z_dim','2048',
      '--fedfed_distill_rounds','15','--fedfed_distill_local_epoch','1','--fedfed_distill_optimizer','adamw','--fedfed_distill_lr','0.001',
      '--fedfed_distill_weight_decay','0.000001','--fedfed_lambda_recon','0.2','--fedfed_lambda_fd','1.5','--fedfed_lambda_align','0.3',
      '--fedfed_lambda_logit_align','0.0','--fedfed_logit_align_temperature','2.0','--fedfed_beta_kl','0.005','--fedfed_lambda_x_ce','0.4',
      '--fedfed_use_augmentation','true','--fedfed_mixup_alpha','2.0','--fedfed_mosaic_batch_size','32','--fedfed_upload_per_class','0',
      '--fedfed_upload_per_client','0','--fedfed_shared_buffer_size','0','--fedfed_shared_per_class_size','0','--fedfed_shared_batch_size','0',
      '--fedfed_shared_cpu_float16','true','--fedfed_noise_type','none','--fedfed_noise_std1','0.0','--fedfed_noise_std2','0.0',
      '--fedfed_distill_ce_on_noisy_xs','false'
    )
  }
  \$started = Get-Date
  \$env:PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:128'
  & \$pythonPath '-u' 'main.py' @args 1> \$stdoutPath 2> \$stderrPath
  if (\$LASTEXITCODE -ne 0) { throw "formal failed: \$runId" }
  \$metricsPath = Find-Metrics \$runId
  if (\$null -eq \$metricsPath) { throw "metrics not found: \$runId" }
  \$m = Parse-Metrics \$metricsPath
  \$duration = [Math]::Round(((Get-Date) - \$started).TotalMinutes, 2)
  \$row = [ordered]@{kind='formal';run_id=\$runId;method=\$method;plugin_name=\$pluginName;alpha=\$alpha;local_epoch=\$localEpoch;duration_min=\$duration;final_acc=\$m.final_acc;best_acc=\$m.best_acc;best_round=\$m.best_round;final_round=\$m.final_round;final_loss=\$m.final_loss;best_loss=\$m.best_loss;metrics_path=\$metricsPath}
  [void]\$rows.Add((New-Object PSObject -Property \$row))
  Write-Summary
  [ordered]@{run_id=\$runId; status='succeeded'; kind='formal'; finished_utc=(Get-Date).ToUniversalTime().ToString('o'); metrics_path=\$metricsPath} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 \$statusPath
}

try {
  New-Item -ItemType Directory -Force -Path \$runsRoot | Out-Null
  Write-QueueStatus 'waiting' '' 'Queue submitted.'
  Wait-PreviousRun
  Write-QueueStatus 'running' '' 'F0/G1/G2 diagnostics starting.'
  Run-Diagnostic 'F0_best_feature_align' 1.5 0.3 0.0 2.0
  Run-Diagnostic 'G1_logit0p5_T2' 1.5 0.0 0.5 2.0
  Run-Diagnostic 'G2_logit1p0_T2' 1.5 0.0 1.0 2.0
  Write-QueueStatus 'running' '' 'Formal FedAvg vs FedFed runs starting.'
  Run-Formal 'FedAvg' 'none' 0.1 1
  Run-Formal 'FedFedPlugin' 'fedfed_image' 0.1 1
  Run-Formal 'FedAvg' 'none' 0.1 5
  Run-Formal 'FedFedPlugin' 'fedfed_image' 0.1 5
  Write-Summary
  New-Item -ItemType Directory -Force -Path \$figuresDir | Out-Null
  & \$pythonPath \$plotScript \$summaryJsonPath \$figuresDir
  Write-QueueStatus 'succeeded' '' 'All diagnostics and formal runs finished.'
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
scp_to_remote "${PLOT_FILE}" "${REMOTE_PLOT_PATH}"
ssh_remote "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${REMOTE_LAUNCHER_PATH}\""
echo "FedFed final diagnostic/formal queue submitted: ${PREFIX}"
