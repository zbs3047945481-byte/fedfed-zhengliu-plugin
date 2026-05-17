#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OFFICIAL_DIR="${REPO_ROOT}/reference_implementations/FedFed_official"
CONFIG_FILE="${SCRIPT_DIR}/official_fedfed_cifar10_a0p1_e1.yaml"
RUN_ID="${RUN_ID:-official_fedfed_cifar10_a0p1_e1_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/remote_results/official_fedfed_reference}"
RUN_DIR="${OUT_ROOT}/${RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${RUN_DIR}"
cp "${CONFIG_FILE}" "${RUN_DIR}/config.yaml"
git -C "${OFFICIAL_DIR}" rev-parse HEAD > "${RUN_DIR}/upstream_commit.txt"

(
  cd "${OFFICIAL_DIR}"
  WANDB_MODE=dryrun "${PYTHON_BIN}" main.py --config_file "${CONFIG_FILE}"
) > "${RUN_DIR}/stdout.log" 2> "${RUN_DIR}/stderr.log"

"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
import ast
import csv
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
stdout = (run_dir / "stdout.log").read_text(errors="ignore").splitlines()
acc_values = []
for line in stdout:
    stripped = line.strip()
    try:
        if re.fullmatch(r"[0-9]*\.?[0-9]+", stripped):
            val = float(stripped)
            if 0.0 <= val <= 1.0:
                acc_values.append(val)
            continue
    except ValueError:
        pass
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            values = ast.literal_eval(stripped)
        except Exception:
            continue
        if isinstance(values, list):
            acc_values = [float(v) for v in values if 0.0 <= float(v) <= 1.0]

summary = {
    "status": "succeeded" if acc_values else "unknown",
    "num_eval_points": len(acc_values),
    "best_acc": max(acc_values) if acc_values else None,
    "final_acc": acc_values[-1] if acc_values else None,
    "best_round": acc_values.index(max(acc_values)) if acc_values else None,
    "final_round": len(acc_values) - 1 if acc_values else None,
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary))
    writer.writeheader()
    writer.writerow(summary)
PY

echo "Official FedFed run finished: ${RUN_DIR}"
