#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

SUITE_PATH="${SCRIPT_DIR}/../experiments/fedfed_thesis_suite.json"
PREFIX="fedfed_thesis_suite_$(date -u +%Y%m%dT%H%M%SZ)"
GROUPS=""
EXPERIMENTS=""
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE_PATH="$2"
      shift 2
      ;;
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --groups)
      GROUPS="$2"
      shift 2
      ;;
    --experiments)
      EXPERIMENTS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT

RUNS_JSON="${TMP_DIR}/suite_runs.json"
QUEUE_FILE="${TMP_DIR}/thesis_suite.ps1"
LAUNCHER_FILE="${TMP_DIR}/thesis_suite_launcher.ps1"
REMOTE_RUNS_JSON="${REMOTE_Codex_DIR}/thesis_suite_runs_${PREFIX}.json"
REMOTE_QUEUE_PATH="${REMOTE_Codex_DIR}/thesis_suite_${PREFIX}.ps1"
REMOTE_LAUNCHER_PATH="${REMOTE_Codex_DIR}/thesis_suite_launcher_${PREFIX}.ps1"
REMOTE_QUEUE_PS_PATH="${REMOTE_QUEUE_PATH//\//\\}"

python3 - "${SUITE_PATH}" "${RUNS_JSON}" "${PREFIX}" "${GROUPS}" "${EXPERIMENTS}" <<'PY'
import json
import sys
from pathlib import Path

suite_path, out_path, prefix, groups_raw, experiments_raw = sys.argv[1:6]
suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
groups = {item.strip() for item in groups_raw.split(",") if item.strip()}
experiments = {item.strip() for item in experiments_raw.split(",") if item.strip()}

def stringify(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

def arg_list(args):
    output = []
    for key, value in args.items():
        output.extend([f"--{key}", stringify(value)])
    return output

rows = []
for exp in suite["experiments"]:
    if exp.get("type") in {"diagnostic", "analysis"}:
        continue
    if groups and exp.get("group") not in groups:
        continue
    if experiments and exp.get("id") not in experiments:
        continue
    grid = exp.get("grid") or [{}]
    for grid_item in grid:
        for run in exp.get("runs", []):
            method = run["method"]
            args = dict(suite["common_args"])
            args.update(grid_item)
            args["plugin_name"] = run["plugin_name"]
            if run["plugin_name"] == "fedfed_image":
                args.update(suite["fedfed_args"])
            args.update(run.get("overrides", {}))
            grid_tag_parts = []
            for key in sorted(grid_item):
                grid_tag_parts.append(f"{key.replace('dirichlet_', '').replace('local_', '')}{str(grid_item[key]).replace('.', 'p')}")
            grid_tag = "_" + "_".join(grid_tag_parts) if grid_tag_parts else ""
            run_id = f"{prefix}_{exp['id']}{grid_tag}_{method}"
            args["experiment_tag"] = run_id
            rows.append({
                "run_id": run_id,
                "experiment_id": exp["id"],
                "group": exp["group"],
                "method": method,
                "plugin_name": run["plugin_name"],
                "arguments": arg_list(args),
            })

Path(out_path).write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"expanded_runs={len(rows)}")
for row in rows:
    print(row["run_id"])
PY

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry run only. Expanded run list saved at ${RUNS_JSON}"
  exit 0
fi

cat > "${QUEUE_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$pythonPath = '${REMOTE_PYTHON}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$prefix = '${PREFIX}'
\$runsJsonPath = '${REMOTE_RUNS_JSON}'
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
  return [ordered]@{
    final_acc = [double]\$m.final_test_acc
    best_acc = [double]\$m.best_test_acc
    final_round = [int]\$m.final_round
    final_loss = [double]\$m.final_test_loss
    best_loss = [double]\$m.best_test_loss
  }
}

function Run-One(\$runSpec) {
  \$runId = [string]\$runSpec.run_id
  \$runDir = Join-Path \$runsRoot \$runId
  \$stdoutPath = Join-Path \$runDir 'stdout.log'
  \$stderrPath = Join-Path \$runDir 'stderr.log'
  \$metaPath = Join-Path \$runDir 'run_meta.json'
  \$statusPath = Join-Path \$runDir 'run_status.json'
  New-Item -ItemType Directory -Force -Path \$runDir | Out-Null
  New-Item -ItemType File -Force -Path \$stdoutPath | Out-Null
  New-Item -ItemType File -Force -Path \$stderrPath | Out-Null

  \$trainArgs = @(\$runSpec.arguments)
  \$startedUtc = (Get-Date).ToUniversalTime().ToString('o')
  [ordered]@{
    run_id = \$runId
    experiment_id = \$runSpec.experiment_id
    group = \$runSpec.group
    method = \$runSpec.method
    plugin_name = \$runSpec.plugin_name
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

  Write-QueueStatus 'running' \$runId "Running \$runId"
  Set-Location \$projectPath
  \$env:PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:128'
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
    Write-QueueStatus 'failed' \$runId 'metrics.json not found.'
    throw "metrics.json not found: \$runId"
  }
  \$metrics = Parse-Metrics \$metricsPath
  \$durationMin = [Math]::Round(((Get-Date \$finishedUtc) - (Get-Date \$startedUtc)).TotalMinutes, 2)
  \$row = [ordered]@{
    run_id = \$runId
    experiment_id = \$runSpec.experiment_id
    group = \$runSpec.group
    method = \$runSpec.method
    plugin_name = \$runSpec.plugin_name
    status = \$finalStatus
    duration_min = \$durationMin
    final_acc = \$metrics.final_acc
    best_acc = \$metrics.best_acc
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
  \$runSpecs = Get-Content -Raw -LiteralPath \$runsJsonPath | ConvertFrom-Json
  Write-QueueStatus 'starting' '' ("Thesis suite starting with {0} runs." -f \$runSpecs.Count)
  foreach (\$runSpec in \$runSpecs) {
    Run-One \$runSpec
  }
  Write-QueueStatus 'succeeded' '' 'Thesis suite finished.'
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
scp_to_remote "${RUNS_JSON}" "${REMOTE_RUNS_JSON}"
scp_to_remote "${QUEUE_FILE}" "${REMOTE_QUEUE_PATH}"
scp_to_remote "${LAUNCHER_FILE}" "${REMOTE_LAUNCHER_PATH}"
ssh_remote "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${REMOTE_LAUNCHER_PATH}\""
echo "FedFed thesis suite submitted: ${PREFIX}"
