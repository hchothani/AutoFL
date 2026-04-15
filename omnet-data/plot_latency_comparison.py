#!/usr/bin/env python
"""Generate comparison plots between baseline and latency-aware AutoFL runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


@dataclass
class RunMetrics:
    aggregate: pd.DataFrame
    server: pd.DataFrame
    name: str


def load_run_metrics(path: Path, name: str) -> RunMetrics:
    aggregate_path = path / "aggregate_metrics.csv"
    server_path = path / "server_round_metrics.csv"
    missing = [p.name for p in (aggregate_path, server_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing expected files in {path}: {', '.join(missing)}"
        )

    aggregate = pd.read_csv(aggregate_path)
    server = pd.read_csv(server_path)
    aggregate["round"] = aggregate["round"].astype(int)
    server["round"] = server["round"].astype(int)
    aggregate["run"] = name
    server["run"] = name
    return RunMetrics(aggregate=aggregate, server=server, name=name)


def prepare_accuracy_dataframe(runs: Dict[str, RunMetrics]) -> pd.DataFrame:
    frames = []
    for metrics in runs.values():
        eval_df = metrics.aggregate[metrics.aggregate["phase"] == "eval"].copy()
        if "global/average/accuracy" not in eval_df.columns:
            continue
        subset = eval_df[["round", "global/average/accuracy", "run"]].rename(
            columns={"global/average/accuracy": "accuracy"}
        )
        frames.append(subset)
    if not frames:
        raise RuntimeError(
            "No evaluation accuracy metrics found. Ensure aggregate_metrics.csv contains eval rows."
        )
    return pd.concat(frames, ignore_index=True)


def prepare_client_timing_dataframe(runs: Dict[str, RunMetrics]) -> pd.DataFrame:
    frames = []
    for metrics in runs.values():
        fit_df = metrics.aggregate[metrics.aggregate["phase"] == "fit"].copy()
        if "local/average/round_time_s" not in fit_df.columns:
            continue
        subset = fit_df[
            [
                "round",
                "local/average/round_time_s",
                "local/average/network_time_s",
                "local/average/training_time_s",
                "run",
            ]
        ]
        subset.rename(
            columns={
                "local/average/round_time_s": "round_time_s",
                "local/average/network_time_s": "network_time_s",
                "local/average/training_time_s": "training_time_s",
            },
            inplace=True,
        )
        frames.append(subset)
    if not frames:
        raise RuntimeError("No fit timing metrics present in aggregate_metrics.csv.")
    return pd.concat(frames, ignore_index=True)


def prepare_server_timing_dataframe(runs: Dict[str, RunMetrics]) -> pd.DataFrame:
    frames = []
    for metrics in runs.values():
        subset = metrics.server[["round", "aggregation_time_s", "run"]].copy()
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def plot_accuracy(df: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="round", y="accuracy", hue="run", marker="o")
    plt.title("Global Accuracy vs Rounds")
    plt.ylabel("Accuracy")
    plt.xlabel("Round")
    plt.legend(title="Run")
    plt.tight_layout()
    plt.savefig(output, dpi=250)
    plt.close()


def plot_runtime(
    client_df: pd.DataFrame, server_df: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    sns.lineplot(
        data=server_df,
        x="round",
        y="aggregation_time_s",
        hue="run",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Server Aggregation Time")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Time [s]")

    sns.lineplot(
        data=client_df, x="round", y="round_time_s", hue="run", marker="o", ax=axes[1]
    )
    axes[1].set_title("Average Client Round Time")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Time [s]")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend().set_title("Run")
    axes[1].legend(handles, labels, title="Run")

    fig.suptitle("Runtime Comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output, dpi=250)
    plt.close(fig)


def write_summary(runs: Dict[str, RunMetrics], output: Path) -> None:
    rows = []
    for name, metrics in runs.items():
        eval_df = metrics.aggregate[metrics.aggregate["phase"] == "eval"]
        fit_df = metrics.aggregate[metrics.aggregate["phase"] == "fit"]
        row = {
            "run": name,
            "eval_accuracy_mean": (
                eval_df["global/average/accuracy"].mean()
                if "global/average/accuracy" in eval_df
                else float("nan")
            ),
            "fit_round_time_mean": (
                fit_df["local/average/round_time_s"].mean()
                if "local/average/round_time_s" in fit_df
                else float("nan")
            ),
            "server_aggregation_time_mean": metrics.server["aggregation_time_s"].mean(),
            "drop_rate": _estimate_drop_rate(metrics.server),
        }
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output, index=False)


def _estimate_drop_rate(server_df: pd.DataFrame) -> float:
    if "dropped_clients" not in server_df.columns or server_df.empty:
        return 0.0
    dropped_counts = server_df["dropped_clients"].fillna("")
    total_rounds = len(server_df)
    total_dropped = 0
    for entry in dropped_counts:
        if not entry:
            continue
        total_dropped += len(str(entry).split(","))
    if total_rounds == 0:
        return 0.0
    return total_dropped / total_rounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and latency-aware AutoFL runs"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline run output directory",
    )
    parser.add_argument(
        "--latency",
        type=Path,
        required=True,
        help="Path to latency-injected run output directory",
    )
    parser.add_argument(
        "--output", type=Path, help="Directory to write comparison artefacts"
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="Baseline",
        help="Display name for the baseline run",
    )
    parser.add_argument(
        "--latency-name",
        type=str,
        default="Latency Injected",
        help="Display name for the latency run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output or (args.latency.parent / "comparison_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        args.baseline_name: load_run_metrics(args.baseline, args.baseline_name),
        args.latency_name: load_run_metrics(args.latency, args.latency_name),
    }

    accuracy_df = prepare_accuracy_dataframe(runs)
    client_df = prepare_client_timing_dataframe(runs)
    server_df = prepare_server_timing_dataframe(runs)

    plot_accuracy(accuracy_df, output_dir / "accuracy_vs_rounds.png")
    plot_runtime(client_df, server_df, output_dir / "runtime_comparison.png")
    write_summary(runs, output_dir / "comparison_summary.csv")

    print(f"Comparison plots written to {output_dir}")


if __name__ == "__main__":
    main()
