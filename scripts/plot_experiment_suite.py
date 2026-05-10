#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


PALETTE = {
    "FedAvg": "#3b5b75",
    "FedFedPlugin": "#c8583e",
    "FedFedFullPool": "#c8583e",
    "FedFedSmallPool": "#7b9e87",
    "FedFedMediumPool": "#d69c4e",
    "FedFedNoNoise": "#777777",
    "FedFedGaussianLow": "#7b9e87",
    "FedFedGaussianPaper": "#c8583e",
    "FedFedGaussianHigh": "#8c5fbf",
    "FedFedLaplacePaper": "#4d8ac8",
}


def setup_style():
    if sns is not None:
        sns.set_theme(
            context="paper",
            style="whitegrid",
            font_scale=1.05,
            rc={
                "axes.edgecolor": "#30343f",
                "grid.color": "#d7dce2",
                "grid.linestyle": "--",
                "axes.facecolor": "#ffffff",
                "figure.facecolor": "#ffffff",
                "savefig.facecolor": "#ffffff",
            },
        )
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def load_suite(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_metrics(results_root):
    root = Path(results_root)
    candidates = list(root.rglob("metrics.json")) + list(root.rglob("live_metrics.json"))
    seen = set()
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        exp_dir = path.parent
        if exp_dir in seen:
            continue
        metrics_path = exp_dir / "metrics.json"
        live_path = exp_dir / "live_metrics.json"
        chosen = metrics_path if metrics_path.exists() else live_path
        seen.add(exp_dir)
        try:
            with open(chosen, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            continue
        yield chosen, metrics


def infer_method(options, plugin_name):
    tag = str(options.get("experiment_tag", ""))
    parts = tag.split("_")
    if parts:
        last = parts[-1]
        known = {
            "fedavg": "FedAvg",
            "fedfed": "FedFedPlugin",
            "fedfedplugin": "FedFedPlugin",
        }
        if last.lower() in known:
            return known[last.lower()]
    if plugin_name == "none":
        return "FedAvg"
    return "FedFedPlugin"


def infer_group(options, method):
    tag = str(options.get("experiment_tag", "")).lower()
    if "shared" in tag or "pool" in tag:
        return "shared_pool"
    if "noise" in tag or str(options.get("fedfed_noise_type", "")).lower() in {"none", "laplace"}:
        return "dp_noise"
    if "distill" in tag:
        return "distillation"
    if "epoch" in tag or "ep" in tag:
        return "local_epoch"
    return "main"


def build_summary(results_root):
    rows = []
    for path, metrics in iter_metrics(results_root):
        options = metrics.get("options", {})
        plugin_name = metrics.get("plugin_name") or options.get("plugin_name", "none")
        method = infer_method(options, plugin_name)
        rounds = metrics.get("rounds") or list(range(len(metrics.get("acc_on_g_test_data", []))))
        acc = metrics.get("acc_on_g_test_data", [])
        best_acc = metrics.get("best_test_acc")
        final_acc = metrics.get("final_test_acc")
        best_round = None
        if acc:
            best_round = rounds[max(range(len(acc)), key=lambda i: acc[i])]
        rows.append({
            "metrics_path": str(path),
            "experiment_tag": options.get("experiment_tag", path.parent.name),
            "group": infer_group(options, method),
            "method": method,
            "plugin_name": plugin_name,
            "dataset": metrics.get("dataset", options.get("dataset_name", "")),
            "alpha": float(options.get("dirichlet_alpha", "nan")),
            "local_epoch": int(options.get("local_epoch", 0)),
            "clients": int(options.get("num_of_clients", 0)),
            "c_fraction": float(options.get("c_fraction", 0.0)),
            "round_num": int(options.get("round_num", 0)),
            "final_round": metrics.get("final_round") if metrics.get("final_round") is not None else (rounds[-1] if rounds else None),
            "best_round": best_round,
            "best_acc": best_acc,
            "final_acc": final_acc,
            "noise_type": options.get("fedfed_noise_type", ""),
            "noise_std1": float(options.get("fedfed_noise_std1", 0.0) or 0.0),
            "noise_std2": float(options.get("fedfed_noise_std2", 0.0) or 0.0),
            "shared_buffer_size": int(options.get("fedfed_shared_buffer_size", 0) or 0),
            "shared_per_class_size": int(options.get("fedfed_shared_per_class_size", 0) or 0),
            "distill_rounds": int(options.get("fedfed_distill_rounds", 0) or 0),
        })
    return pd.DataFrame(rows)


def load_curve(row):
    with open(row["metrics_path"], "r", encoding="utf-8") as f:
        metrics = json.load(f)
    rounds = metrics.get("rounds") or list(range(len(metrics.get("acc_on_g_test_data", []))))
    return pd.DataFrame({
        "round": rounds,
        "accuracy": metrics.get("acc_on_g_test_data", []),
        "method": row["method"],
        "alpha": row["alpha"],
        "local_epoch": row["local_epoch"],
        "experiment_tag": row["experiment_tag"],
    })


def save_fig(fig, output_dir, name):
    png = Path(output_dir) / f"{name}.png"
    pdf = Path(output_dir) / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_main_curves(summary, output_dir):
    subset = summary[(summary["alpha"] == 0.1) & (summary["local_epoch"] == 1)]
    if subset.empty:
        return
    curves = [load_curve(row) for _, row in subset.iterrows()]
    if not curves:
        return
    data = pd.concat(curves, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for method, part in data.groupby("method"):
        color = PALETTE.get(method)
        ax.plot(part["round"], part["accuracy"] * 100.0, label=method, linewidth=2.2, color=color)
    ax.set_title("FedAvg vs FedFed Plugin on CIFAR-10")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(frameon=False)
    save_fig(fig, output_dir, "main_accuracy_curve")


def plot_best_final_bar(summary, output_dir):
    subset = summary[(summary["alpha"] == 0.1) & (summary["local_epoch"] == 1)]
    if subset.empty:
        return
    data = subset.melt(
        id_vars=["method"],
        value_vars=["best_acc", "final_acc"],
        var_name="metric",
        value_name="accuracy",
    )
    data["metric"] = data["metric"].map({"best_acc": "Best", "final_acc": "Final"})
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if sns is not None:
        sns.barplot(data=data, x="method", y=data["accuracy"] * 100.0, hue="metric", ax=ax, palette=["#3b5b75", "#c8583e"])
    else:
        data.pivot(index="method", columns="metric", values="accuracy").mul(100).plot(kind="bar", ax=ax)
    ax.set_title("Best and Final Accuracy")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, title="")
    save_fig(fig, output_dir, "main_best_final_bar")


def plot_alpha_gain(summary, output_dir):
    required = summary[summary["method"].isin(["FedAvg", "FedFedPlugin"])]
    if required.empty:
        return
    pivot = required.pivot_table(index="alpha", columns="method", values="best_acc", aggfunc="max").reset_index()
    if not {"FedAvg", "FedFedPlugin"}.issubset(set(pivot.columns)):
        return
    pivot["gain"] = (pivot["FedFedPlugin"] - pivot["FedAvg"]) * 100.0
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(pivot["alpha"], pivot["gain"], marker="o", linewidth=2.2, color="#c8583e")
    ax.set_title("Accuracy Gain under Different Heterogeneity")
    ax.set_xlabel("Dirichlet alpha")
    ax.set_ylabel("Best accuracy gain (pp)")
    ax.invert_xaxis()
    save_fig(fig, output_dir, "alpha_gain")


def plot_noise(summary, output_dir):
    data = summary[summary["plugin_name"] == "fedfed_image"].copy()
    data = data[data["noise_type"].astype(str) != ""]
    if data.empty:
        return
    data["noise_label"] = data.apply(
        lambda r: f"{r['noise_type']} {r['noise_std1']:.2f}/{r['noise_std2']:.2f}",
        axis=1,
    )
    data["noise_level"] = (data["noise_std1"] + data["noise_std2"]) / 2.0
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for noise_type, part in data.groupby("noise_type"):
        part = part.sort_values("noise_level")
        ax.plot(part["noise_level"], part["best_acc"] * 100.0, marker="o", linewidth=2.0, label=noise_type)
    ax.set_title("DP Noise vs Accuracy")
    ax.set_xlabel("Average noise level")
    ax.set_ylabel("Best accuracy (%)")
    ax.legend(frameon=False)
    save_fig(fig, output_dir, "dp_noise_accuracy")


def plot_shared_pool(summary, output_dir):
    data = summary[summary["plugin_name"] == "fedfed_image"].copy()
    if data.empty:
        return
    data["pool_label"] = data.apply(
        lambda r: "Full" if r["shared_buffer_size"] == 0 and r["shared_per_class_size"] == 0 else str(r["shared_buffer_size"]),
        axis=1,
    )
    pool_data = data[data["pool_label"].isin(["800", "4000", "Full"])]
    if pool_data.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    if sns is not None:
        sns.barplot(data=pool_data, x="pool_label", y=pool_data["best_acc"] * 100.0, ax=ax, color="#c8583e")
    else:
        ax.bar(pool_data["pool_label"], pool_data["best_acc"] * 100.0, color="#c8583e")
    ax.set_title("Shared Sensitive Feature Pool Size")
    ax.set_xlabel("Shared buffer size")
    ax.set_ylabel("Best accuracy (%)")
    save_fig(fig, output_dir, "shared_pool_best_acc")


def plot_distillation(summary, output_dir):
    data = summary[(summary["plugin_name"] == "fedfed_image") & (summary["distill_rounds"] > 0)].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.scatter(data["distill_rounds"], data["best_acc"] * 100.0, s=70, color="#c8583e")
    ax.plot(data.sort_values("distill_rounds")["distill_rounds"], data.sort_values("distill_rounds")["best_acc"] * 100.0, color="#c8583e", linewidth=1.8)
    ax.set_title("Distillation Rounds vs Accuracy")
    ax.set_xlabel("Feature distillation rounds")
    ax.set_ylabel("Best accuracy (%)")
    save_fig(fig, output_dir, "distill_rounds_best_acc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="experiments/fedfed_thesis_suite.json")
    parser.add_argument("--results-root", default="result")
    parser.add_argument("--output-dir", default="figures/fedfed_thesis_suite")
    args = parser.parse_args()

    setup_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.suite).exists():
        suite = load_suite(args.suite)
        with open(output_dir / "suite_used.json", "w", encoding="utf-8") as f:
            json.dump(suite, f, indent=2)

    summary = build_summary(args.results_root)
    if summary.empty:
        raise SystemExit(f"No metrics found under {args.results_root}")

    summary = summary.sort_values(["group", "alpha", "local_epoch", "method", "experiment_tag"])
    summary.to_csv(output_dir / "summary.csv", index=False)

    plot_main_curves(summary, output_dir)
    plot_best_final_bar(summary, output_dir)
    plot_alpha_gain(summary, output_dir)
    plot_noise(summary, output_dir)
    plot_shared_pool(summary, output_dir)
    plot_distillation(summary, output_dir)
    print(f"Wrote summary and figures to {output_dir}")


if __name__ == "__main__":
    main()
