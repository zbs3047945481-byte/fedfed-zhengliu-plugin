#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def stringify(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def arg_list(args):
    output = []
    for key, value in args.items():
        output.extend([f"--{key}", stringify(value)])
    return output


def expand_runs(suite, prefix, skip_groups):
    rows = []
    for exp in suite["experiments"]:
        if exp.get("type") in {"diagnostic", "analysis"}:
            continue
        if exp.get("group") in skip_groups or exp.get("id") in skip_groups:
            continue
        grid = exp.get("grid") or [{}]
        for grid_item in grid:
            for run in exp.get("runs", []):
                args = dict(suite["common_args"])
                args.update(exp.get("overrides", {}))
                args.update(grid_item)
                args["plugin_name"] = run["plugin_name"]
                if run["plugin_name"] == "fedfed_image":
                    args.update(suite["fedfed_args"])
                args.update(run.get("overrides", {}))
                grid_tag = ""
                if grid_item:
                    parts = []
                    for key in sorted(grid_item):
                        clean = key.replace("dirichlet_", "").replace("local_", "")
                        parts.append(f"{clean}{str(grid_item[key]).replace('.', 'p')}")
                    grid_tag = "_" + "_".join(parts)
                run_id = f"{prefix}_{exp['id']}{grid_tag}_{run['method']}"
                args["experiment_tag"] = run_id
                rows.append({
                    "run_id": run_id,
                    "experiment_id": exp["id"],
                    "group": exp["group"],
                    "method": run["method"],
                    "plugin_name": run["plugin_name"],
                    "arguments": arg_list(args),
                    "options": args,
                })
    return rows


def expand_diagnostics(suite, prefix, skip_groups):
    rows = []
    for exp in suite["experiments"]:
        if exp.get("type") != "diagnostic":
            continue
        if exp.get("group") in skip_groups or exp.get("id") in skip_groups:
            continue
        args = dict(suite["common_args"])
        args.update(suite.get("fedfed_args", {}))
        args["plugin_name"] = "fedfed_image"
        args["experiment_tag"] = f"{prefix}_{exp['id']}"
        rows.append({
            "run_id": args["experiment_tag"],
            "experiment_id": exp["id"],
            "group": exp["group"],
            "method": "FedFedDiagnostic",
            "plugin_name": "fedfed_image",
            "arguments": arg_list(args),
            "options": args,
        })
    return rows


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_with_keys(path, rows, keys=None):
    if not rows:
        return
    if keys is None:
        keys = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_metrics(project_dir, run_id, dataset_name):
    root = project_dir / "result" / str(dataset_name)
    if not root.exists():
        return None
    candidates = sorted(root.glob(f"*{run_id}*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_training_spec(args, project_dir, runs_root, spec, index, total):
    run_id = spec["run_id"]
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_meta.json", spec)
    write_json(run_dir / "run_status.json", {"run_id": run_id, "status": "running", "index": index, "total": total})
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    start = time.time()
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as out, (run_dir / "stderr.log").open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            [args.python, "-u", "main.py", *spec["arguments"]],
            cwd=str(project_dir),
            stdout=out,
            stderr=err,
            env=env,
        )
    duration = round((time.time() - start) / 60.0, 2)
    if proc.returncode != 0:
        write_json(run_dir / "run_status.json", {"run_id": run_id, "status": "failed", "exit_code": proc.returncode})
        raise RuntimeError(f"{run_id} failed with exit code {proc.returncode}")
    metrics_path = find_metrics(project_dir, run_id, spec["options"].get("dataset_name", "cifar10"))
    if metrics_path is None:
        raise RuntimeError(f"metrics not found for {run_id}")
    metrics = parse_metrics(metrics_path)
    row = {
        "run_id": run_id,
        "experiment_id": spec["experiment_id"],
        "group": spec["group"],
        "method": spec["method"],
        "plugin_name": spec["plugin_name"],
        "status": "succeeded",
        "duration_min": duration,
        "dirichlet_alpha": spec["options"].get("dirichlet_alpha"),
        "local_epoch": spec["options"].get("local_epoch"),
        "batch_size": spec["options"].get("batch_size"),
        "dataset_name": spec["options"].get("dataset_name"),
        "image_size": spec["options"].get("image_size"),
        "num_classes": spec["options"].get("num_classes"),
        "distill_rounds": spec["options"].get("fedfed_distill_rounds", 0),
        "shared_buffer_size": spec["options"].get("fedfed_shared_buffer_size", 0),
        "shared_per_class_size": spec["options"].get("fedfed_shared_per_class_size", 0),
        **metrics,
    }
    write_json(run_dir / "run_status.json", {"run_id": run_id, "status": "succeeded", "duration_min": duration, "metrics_path": str(metrics_path)})
    return row


def parse_metrics(path):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    acc = metrics.get("acc_on_g_test_data", [])
    rounds = metrics.get("rounds", list(range(len(acc))))
    best_idx = max(range(len(acc)), key=lambda i: acc[i]) if acc else None
    return {
        "final_acc": metrics.get("final_test_acc"),
        "best_acc": metrics.get("best_test_acc"),
        "best_round": rounds[best_idx] if best_idx is not None and best_idx < len(rounds) else None,
        "final_round": metrics.get("final_round"),
        "final_loss": metrics.get("final_test_loss"),
        "best_loss": metrics.get("best_test_loss"),
        "metrics_path": str(path),
    }


def plot_results(summary_rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import pandas as pd
        import seaborn as sns
    except ModuleNotFoundError:
        return plot_results_basic(summary_rows, output_dir, plt)

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row for row in summary_rows if row.get("status") == "succeeded"])
    if df.empty:
        return
    df.to_csv(output_dir / "summary.csv", index=False)
    plot_df = df.copy()
    main_base = df[df["experiment_id"] == "main_a0p1_e1"].copy()
    if not main_base.empty:
        heter_base = main_base.copy()
        heter_base["experiment_id"] = "heterogeneity_alpha"
        heter_base["group"] = "heterogeneity"
        heter_base["run_id"] = heter_base["run_id"].astype(str) + "_reused_for_alpha0p1"
        local_base = main_base.copy()
        local_base["experiment_id"] = "local_epoch"
        local_base["group"] = "local_epoch"
        local_base["run_id"] = local_base["run_id"].astype(str) + "_reused_for_E1"
        plot_df = pd.concat([plot_df, heter_base, local_base], ignore_index=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)

    # Best/final overview.
    melted = plot_df.melt(
        id_vars=["experiment_id", "group", "method", "run_id"],
        value_vars=["best_acc", "final_acc"],
        var_name="metric",
        value_name="accuracy",
    )
    melted["accuracy"] = melted["accuracy"].astype(float) * 100.0
    for group, part in melted.groupby("group"):
        fig, ax = plt.subplots(figsize=(max(7.5, len(part["run_id"].unique()) * 0.55), 4.8))
        sns.barplot(data=part, x="run_id", y="accuracy", hue="metric", ax=ax)
        ax.set_title(f"{group}: best/final accuracy")
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy (%)")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.legend(frameon=False, title="")
        fig.tight_layout()
        fig.savefig(output_dir / f"{group}_best_final.png", dpi=300)
        fig.savefig(output_dir / f"{group}_best_final.pdf")
        plt.close(fig)

    # Alpha gain.
    alpha_df = plot_df[plot_df["experiment_id"] == "heterogeneity_alpha"].copy()
    if not alpha_df.empty:
        pivot = alpha_df.pivot_table(index="dirichlet_alpha", columns="method", values="best_acc", aggfunc="max")
        if {"FedAvg", "FedFedPlugin"}.issubset(set(pivot.columns)):
            pivot = pivot.reset_index()
            pivot["gain_pp"] = (pivot["FedFedPlugin"] - pivot["FedAvg"]) * 100.0
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            sns.lineplot(data=pivot, x="dirichlet_alpha", y="gain_pp", marker="o", ax=ax)
            ax.invert_xaxis()
            ax.set_title("FedFed gain under label/quantity skew")
            ax.set_xlabel("Dirichlet alpha")
            ax.set_ylabel("Best accuracy gain (pp)")
            fig.tight_layout()
            fig.savefig(output_dir / "heterogeneity_alpha_gain.png", dpi=300)
            fig.savefig(output_dir / "heterogeneity_alpha_gain.pdf")
            plt.close(fig)

    # Curves for all experiment groups.
    for exp_id in sorted(set(plot_df["experiment_id"].astype(str))):
        part = plot_df[plot_df["experiment_id"] == exp_id]
        if part.empty:
            continue
        for _, row in part.iterrows():
            metrics = json.loads(Path(row["metrics_path"]).read_text(encoding="utf-8"))
            rounds = metrics.get("rounds", [])
            acc = metrics.get("acc_on_g_test_data", [])
            if not rounds or not acc:
                continue
            label = f"{row['method']} a={row['dirichlet_alpha']} E={row['local_epoch']}"
            plt.plot(rounds, [v * 100 for v in acc], linewidth=1.8, label=label)
        if plt.gca().lines:
            fig = plt.gcf()
            fig.set_size_inches(8.2, 4.8)
            plt.title(f"{exp_id}: accuracy curves")
            plt.xlabel("Round")
            plt.ylabel("Test accuracy (%)")
            plt.legend(frameon=False, fontsize=8)
            plt.grid(True, linestyle="--", alpha=0.35)
            fig.tight_layout()
            fig.savefig(output_dir / f"{exp_id}_curves.png", dpi=300)
            fig.savefig(output_dir / f"{exp_id}_curves.pdf")
            plt.close(fig)

    # Cross-domain summary when a suite spans multiple datasets.
    if "dataset_name" in df.columns and df["dataset_name"].nunique() > 1:
        cross_df = df.copy()
        cross_df["best_acc_pct"] = cross_df["best_acc"].astype(float) * 100.0
        cross_df["final_acc_pct"] = cross_df["final_acc"].astype(float) * 100.0
        for metric, ylabel, filename in [
            ("best_acc_pct", "Best accuracy (%)", "cross_domain_best_acc_by_dataset"),
            ("final_acc_pct", "Final accuracy (%)", "cross_domain_final_acc_by_dataset"),
        ]:
            fig, ax = plt.subplots(figsize=(7.4, 4.6))
            sns.barplot(data=cross_df, x="dataset_name", y=metric, hue="method", ax=ax)
            ax.set_title(ylabel.replace(" (%)", " by dataset"))
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False, title="")
            fig.tight_layout()
            fig.savefig(output_dir / f"{filename}.png", dpi=300)
            fig.savefig(output_dir / f"{filename}.pdf")
            plt.close(fig)
        pivot = cross_df.pivot_table(index="dataset_name", columns="method", values="best_acc", aggfunc="max")
        if {"FedAvg", "FedFedPlugin"}.issubset(set(pivot.columns)):
            gain = ((pivot["FedFedPlugin"] - pivot["FedAvg"]) * 100.0).reset_index(name="gain_pp")
            gain.to_csv(output_dir / "cross_domain_gain_by_dataset.csv", index=False)
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
            sns.barplot(data=gain, x="dataset_name", y="gain_pp", ax=ax, color="#4C78A8")
            ax.axhline(0.0, color="#444444", linewidth=1.0)
            ax.set_title("FedFed best-accuracy gain by dataset")
            ax.set_xlabel("")
            ax.set_ylabel("Gain over FedAvg (pp)")
            fig.tight_layout()
            fig.savefig(output_dir / "cross_domain_gain_by_dataset.png", dpi=300)
            fig.savefig(output_dir / "cross_domain_gain_by_dataset.pdf")
            plt.close(fig)


def plot_results_basic(summary_rows, output_dir, plt):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in summary_rows if row.get("status") == "succeeded"]
    if not rows:
        return
    write_csv_with_keys(output_dir / "summary.csv", rows)

    by_group = {}
    for row in rows:
        by_group.setdefault(str(row.get("group", "results")), []).append(row)

    for group, part in by_group.items():
        methods = [str(row.get("method", row.get("run_id", ""))) for row in part]
        x = list(range(len(methods)))
        best = [float(row.get("best_acc") or 0.0) * 100.0 for row in part]
        final = [float(row.get("final_acc") or 0.0) * 100.0 for row in part]
        width = 0.36
        fig, ax = plt.subplots(figsize=(max(7.5, len(methods) * 1.25), 4.8))
        ax.bar([v - width / 2 for v in x], best, width=width, label="best_acc")
        ax.bar([v + width / 2 for v in x], final, width=width, label="final_acc")
        ax.set_title(f"{group}: best/final accuracy")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{group}_best_final.png", dpi=300)
        fig.savefig(output_dir / f"{group}_best_final.pdf")
        plt.close(fig)

    by_exp = {}
    for row in rows:
        by_exp.setdefault(str(row.get("experiment_id", "experiment")), []).append(row)
    for exp_id, part in by_exp.items():
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        has_line = False
        for row in part:
            metrics_path = row.get("metrics_path")
            if not metrics_path:
                continue
            path = Path(metrics_path)
            if not path.exists():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            rounds = metrics.get("rounds", [])
            acc = metrics.get("acc_on_g_test_data", [])
            if not rounds or not acc:
                continue
            label = f"{row.get('method')} a={row.get('dirichlet_alpha')} E={row.get('local_epoch')}"
            ax.plot(rounds, [float(v) * 100.0 for v in acc], linewidth=1.8, label=label)
            has_line = True
        if has_line:
            ax.set_title(f"{exp_id}: accuracy curves")
            ax.set_xlabel("Round")
            ax.set_ylabel("Test accuracy (%)")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / f"{exp_id}_curves.png", dpi=300)
            fig.savefig(output_dir / f"{exp_id}_curves.pdf")
        plt.close(fig)

    datasets = sorted({str(row.get("dataset_name")) for row in rows if row.get("dataset_name")})
    if len(datasets) > 1:
        methods = sorted({str(row.get("method")) for row in rows if row.get("method")})
        width = 0.8 / max(len(methods), 1)
        x = list(range(len(datasets)))
        for metric, ylabel, filename in [
            ("best_acc", "Best accuracy (%)", "cross_domain_best_acc_by_dataset"),
            ("final_acc", "Final accuracy (%)", "cross_domain_final_acc_by_dataset"),
        ]:
            fig, ax = plt.subplots(figsize=(7.4, 4.6))
            for offset, method in enumerate(methods):
                values = []
                for dataset in datasets:
                    matches = [
                        row for row in rows
                        if str(row.get("dataset_name")) == dataset and str(row.get("method")) == method
                    ]
                    values.append(float(matches[0].get(metric) or 0.0) * 100.0 if matches else 0.0)
                positions = [v - 0.4 + width / 2 + offset * width for v in x]
                ax.bar(positions, values, width=width, label=method)
            ax.set_title(ylabel.replace(" (%)", " by dataset"))
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(datasets)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / f"{filename}.png", dpi=300)
            fig.savefig(output_dir / f"{filename}.pdf")
            plt.close(fig)

        gain_rows = []
        for dataset in datasets:
            fedavg = next((row for row in rows if str(row.get("dataset_name")) == dataset and row.get("method") == "FedAvg"), None)
            fedfed = next((row for row in rows if str(row.get("dataset_name")) == dataset and row.get("method") == "FedFedPlugin"), None)
            if fedavg and fedfed:
                gain_rows.append({
                    "dataset_name": dataset,
                    "gain_pp": (float(fedfed.get("best_acc") or 0.0) - float(fedavg.get("best_acc") or 0.0)) * 100.0,
                })
        if gain_rows:
            write_csv_with_keys(output_dir / "cross_domain_gain_by_dataset.csv", gain_rows, ["dataset_name", "gain_pp"])
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
            ax.bar([row["dataset_name"] for row in gain_rows], [row["gain_pp"] for row in gain_rows], color="#4C78A8")
            ax.axhline(0.0, color="#444444", linewidth=1.0)
            ax.set_title("FedFed best-accuracy gain by dataset")
            ax.set_ylabel("Gain over FedAvg (pp)")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            fig.tight_layout()
            fig.savefig(output_dir / "cross_domain_gain_by_dataset.png", dpi=300)
            fig.savefig(output_dir / "cross_domain_gain_by_dataset.pdf")
            plt.close(fig)


def plot_diagnostics(diagnostic_rows, output_dir):
    if not diagnostic_rows:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    metric_labels = {
        "x_to_x": "x -> x",
        "xs_to_xs": "xs -> xs",
        "xr_to_xr": "xr -> xr",
        "x_plus_xs_to_x": "x+xs -> x",
    }
    for row in diagnostic_rows:
        if row.get("status") != "succeeded":
            continue
        for metric, label in metric_labels.items():
            value = row.get(metric)
            if value is not None:
                rows.append({"metric": label, "accuracy": float(value) * 100.0})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "diagnostic_feature_bar.csv", index=False)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    sns.barplot(data=df, x="metric", y="accuracy", ax=ax, color="#4C78A8")
    ax.set_title("Feature diagnostic")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(100.0, df["accuracy"].max() + 5.0))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "diagnostic_feature_bar.png", dpi=300)
    fig.savefig(output_dir / "diagnostic_feature_bar.pdf")
    plt.close(fig)


def find_diagnostic_metrics(project_dir, run_id):
    root = project_dir / "result" / "cifar10"
    if not root.exists():
        return None
    candidates = sorted(root.glob(f"fedfed_xs_diagnostic_{run_id}/diagnostic_metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_diagnostic_metrics(path):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    keep = ["x_to_x", "xs_to_xs", "xr_to_xr", "x_plus_xs_to_x"]
    return {key: metrics.get(key) for key in keep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--runs-root", default="/root/runs")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-groups", default="dp_noise")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Maximum number of training runs to execute concurrently.")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    runs_root = Path(args.runs_root)
    prefix = args.prefix
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    skip_groups = {item.strip() for item in args.skip_groups.split(",") if item.strip()}
    specs = expand_runs(suite, prefix, skip_groups)
    diagnostics = expand_diagnostics(suite, prefix, skip_groups)

    status_path = runs_root / f"{prefix}_queue_status.json"
    summary_path = runs_root / f"{prefix}_summary.json"
    summary_csv = runs_root / f"{prefix}_summary.csv"
    figures_dir = runs_root / f"{prefix}_figures"
    specs_path = runs_root / f"{prefix}_expanded_runs.json"
    rows = []
    diagnostic_rows = []
    write_json(specs_path, specs)

    def status(state, current_run, message):
        write_json(status_path, {
            "prefix": prefix,
            "status": state,
            "current_run": current_run,
            "message": message,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary_json": str(summary_path),
            "summary_csv": str(summary_csv),
            "figures_dir": str(figures_dir),
            "num_runs": len(specs),
            "num_diagnostics": len(diagnostics),
        })

    try:
        max_workers = max(int(args.max_workers), 1)
        status("running", "", f"Starting {len(specs)} runs with max_workers={max_workers}")
        indexed_specs = [(index, spec) for index, spec in enumerate(specs, start=1)]
        if max_workers == 1:
            for index, spec in indexed_specs:
                status("running", spec["run_id"], f"Running {index}/{len(specs)}")
                row = run_training_spec(args, project_dir, runs_root, spec, index, len(specs))
                rows.append(row)
                write_json(summary_path, rows)
                write_csv(summary_csv, rows)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_spec = {
                    executor.submit(run_training_spec, args, project_dir, runs_root, spec, index, len(specs)): (index, spec)
                    for index, spec in indexed_specs
                }
                status("running", "", f"Running up to {max_workers}/{len(specs)} concurrently")
                for future in concurrent.futures.as_completed(future_to_spec):
                    index, spec = future_to_spec[future]
                    row = future.result()
                    rows.append(row)
                    write_json(summary_path, rows)
                    write_csv(summary_csv, rows)
                    status("running", spec["run_id"], f"Completed {len(rows)}/{len(specs)}")
        for index, spec in enumerate(diagnostics, start=1):
            run_id = spec["run_id"]
            run_dir = runs_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "run_meta.json", spec)
            write_json(run_dir / "run_status.json", {"run_id": run_id, "status": "running", "index": index, "total": len(diagnostics)})
            status("running", run_id, f"Running diagnostic {index}/{len(diagnostics)}")
            env = os.environ.copy()
            env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
            start = time.time()
            with (run_dir / "stdout.log").open("w", encoding="utf-8") as out, (run_dir / "stderr.log").open("w", encoding="utf-8") as err:
                proc = subprocess.run(
                    [args.python, "-u", "scripts/run_fedfed_xs_diagnostic.py", *spec["arguments"]],
                    cwd=str(project_dir),
                    stdout=out,
                    stderr=err,
                    env=env,
                )
            duration = round((time.time() - start) / 60.0, 2)
            if proc.returncode != 0:
                write_json(run_dir / "run_status.json", {"run_id": run_id, "status": "failed", "exit_code": proc.returncode})
                raise RuntimeError(f"{run_id} diagnostic failed with exit code {proc.returncode}")
            metrics_path = find_diagnostic_metrics(project_dir, run_id)
            if metrics_path is None:
                raise RuntimeError(f"diagnostic metrics not found for {run_id}")
            row = {
                "run_id": run_id,
                "experiment_id": spec["experiment_id"],
                "group": spec["group"],
                "method": spec["method"],
                "plugin_name": spec["plugin_name"],
                "status": "succeeded",
                "duration_min": duration,
                "metrics_path": str(metrics_path),
                **parse_diagnostic_metrics(metrics_path),
            }
            diagnostic_rows.append(row)
            write_json(runs_root / f"{prefix}_diagnostic_summary.json", diagnostic_rows)
            write_csv(runs_root / f"{prefix}_diagnostic_summary.csv", diagnostic_rows)
            write_json(run_dir / "run_status.json", {"run_id": run_id, "status": "succeeded", "duration_min": duration, "metrics_path": str(metrics_path)})
        plot_results(rows, figures_dir)
        plot_diagnostics(diagnostic_rows, figures_dir)
        status("succeeded", "", "Suite finished")
    except Exception as exc:
        write_json(summary_path, rows)
        write_csv(summary_csv, rows)
        status("failed", "", str(exc))
        raise


if __name__ == "__main__":
    main()
