#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_rows(paths):
    rows_by_method = {}
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "succeeded":
                    continue
                method = row.get("method") or row.get("run_id")
                rows_by_method[method] = row
    order = [
        "FedProx",
        "FedProxFedFedPlugin",
        "SCAFFOLD",
        "SCAFFOLDFedFedPlugin",
        "FedAvgM",
        "FedAvgMFedFedPlugin",
        "FedNova",
        "FedNovaFedFedPlugin",
    ]
    return [rows_by_method[key] for key in order if key in rows_by_method]


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_summary(rows, output_dir):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "combined_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "combined_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def plot(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(rows, output_dir)
    labels = [row["method"] for row in rows]
    x = list(range(len(labels)))
    width = 0.36
    best = [as_float(row.get("best_acc")) * 100.0 for row in rows]
    final = [as_float(row.get("final_acc")) * 100.0 for row in rows]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar([i - width / 2 for i in x], best, width=width, label="Best")
    ax.bar([i + width / 2 for i in x], final, width=width, label="Final")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("FedFed plugin across FL backbones")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "combined_best_final.png", dpi=300)
    fig.savefig(output_dir / "combined_best_final.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for row in rows:
        metrics_path = Path(row.get("metrics_path", ""))
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rounds = metrics.get("rounds", [])
        acc = metrics.get("acc_on_g_test_data", [])
        if rounds and acc:
            ax.plot(rounds, [as_float(v) * 100.0 for v in acc], linewidth=1.8, label=row["method"])
    ax.set_xlabel("Round")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy curves across FL backbones")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "combined_accuracy_curves.png", dpi=300)
    fig.savefig(output_dir / "combined_accuracy_curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, help="Input summary CSV. Later files override duplicate methods.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    plot(read_rows(args.summary), Path(args.output_dir))


if __name__ == "__main__":
    main()
