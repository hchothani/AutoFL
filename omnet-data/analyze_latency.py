#!/usr/bin/env python
"""for exploring OMNeT++ VoIP latency exports.

The CSV produced by `scavetool export` stores every statistic as a pair of columns:
 one column for the sample timestamp and another for the sample value.  This script
 pulls out the most relevant metrics/

Example usage (from the project root):

    python omnet-data/analyze_latency.py \
        --input omnet-data/latency_with_10cars2RSU_30.09.2025.csv

"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

# Use a non-interactive backend so we can run headless.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

MAX_DEFAULT_CLIENTS = 10
GENERATED_THROUGHPUT_SCALE = (
    8.0  # Convert kBytes/s -> kbits/s so it matches server stats
)


@dataclass(frozen=True)
class MetricLocation:
    """Reference to a metric's time/value column pair."""

    time_idx: int
    value_idx: int
    raw_label: str

    def base_name(self, key: str, suffix: str) -> str:
        return f"{key}_car{self.client_id()}_{suffix}"

    def client_id(self) -> int:
        match = re.search(r"\[(\d+)\]", self.raw_label)
        if not match:
            raise ValueError(f"Unable to infer client id from label: {self.raw_label}")
        return int(match.group(1))


def load_header(path: Path) -> Tuple[List[str], int]:
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    return header, len(header)


def collect_metric_locations(
    header: Sequence[str],
    pattern: str,
    max_clients: int,
) -> Dict[int, MetricLocation]:
    regex = re.compile(pattern)
    mapping: Dict[int, MetricLocation] = {}
    for idx, raw in enumerate(header):
        label = raw.strip()
        if not label:
            continue
        match = regex.search(label)
        if not match:
            continue
        client_id = int(match.group(1))
        if client_id >= max_clients:
            continue
        value_idx = idx + 1
        if value_idx >= len(header):
            continue
        # When the header originates from scavetool the value column is unnamed (empty string),
        # so we simply accept the adjacent column.
        if client_id not in mapping:
            mapping[client_id] = MetricLocation(idx, value_idx, label)
    return mapping


def assemble_metric_map(
    header: Sequence[str],
    max_clients: int,
) -> Dict[str, Dict[int, MetricLocation]]:
    metric_patterns = {
        "frame_delay": r"voIPFrameDelay:vector\s+Highway\.server\.app\[(\d+)\]",
        "received_throughput": r"voIPReceivedThroughput:vector\s+Highway\.server\.app\[(\d+)\]",
        "generated_throughput": r"voIPGeneratedThroughput:vector\s+Highway\.car\[(\d+)\]\.app\[0\]",
        "serving_cell": r"servingCell:vector\s+Highway\.car\[(\d+)\]",
    }

    metric_map: Dict[str, Dict[int, MetricLocation]] = {}
    for key, pattern in metric_patterns.items():
        metric_map[key] = collect_metric_locations(header, pattern, max_clients)

    return metric_map


def build_dataframe(
    path: Path,
    total_columns: int,
    metric_map: Mapping[str, Mapping[int, MetricLocation]],
) -> Tuple[pd.DataFrame, Dict[int, str], Dict[int, str]]:
    index_to_name: Dict[int, str] = {}
    index_to_metric: Dict[int, str] = {}
    usecols = set()

    for metric_key, per_car in metric_map.items():
        for client_id, loc in per_car.items():
            time_name = f"{metric_key}_car{client_id}_time"
            value_name = f"{metric_key}_car{client_id}_value"
            index_to_name[loc.time_idx] = time_name
            index_to_name[loc.value_idx] = value_name
            index_to_metric[loc.time_idx] = metric_key
            index_to_metric[loc.value_idx] = metric_key
            usecols.add(loc.time_idx)
            usecols.add(loc.value_idx)

    if not usecols:
        return pd.DataFrame(), {}, {}

    sorted_usecols = sorted(usecols)
    col_names = [f"col_{i}" for i in range(total_columns)]
    df = pd.read_csv(
        path,
        header=None,
        names=col_names,
        usecols=sorted_usecols,
        skiprows=1,
    )

    # Rename the subset to meaningful labels.
    rename_map = {
        f"col_{idx}": index_to_name[idx]
        for idx in sorted_usecols
        if idx in index_to_name
    }
    df.rename(columns=rename_map, inplace=True)
    return df, index_to_name, index_to_metric


def build_long_form(
    data: pd.DataFrame,
    metric_key: str,
    per_car: Mapping[int, MetricLocation],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for client_id, loc in per_car.items():
        time_col = f"{metric_key}_car{client_id}_time"
        value_col = f"{metric_key}_car{client_id}_value"
        if time_col not in data.columns or value_col not in data.columns:
            continue
        block = data[[time_col, value_col]].dropna()
        if block.empty:
            continue
        block = block.rename(columns={time_col: "time", value_col: "value"})
        block["car_id"] = client_id
        frames.append(block)

    if not frames:
        return pd.DataFrame(columns=["time", "value", "car_id"])

    result = pd.concat(frames, ignore_index=True)
    result = result[(result["time"].notna()) & (result["value"].notna())]
    result.sort_values(["car_id", "time"], inplace=True)
    return result


def compute_frame_delay_stats(
    frame_delay: pd.DataFrame, serving_cell: pd.DataFrame
) -> pd.DataFrame:
    if frame_delay.empty:
        return pd.DataFrame()

    stats = frame_delay.groupby("car_id")["value"].agg(
        sample_count="count",
        mean_delay="mean",
        median_delay="median",
        p95_delay=lambda x: x.quantile(0.95),
        max_delay="max",
    )

    if not serving_cell.empty:
        transitions = (
            serving_cell.sort_values(["car_id", "time"])  # ensure order
            .groupby("car_id")["value"]
            .apply(lambda s: s.dropna().astype(float).diff().fillna(0).ne(0).sum())
        )
        stats["cell_transitions"] = transitions

    return stats


def downsample(group: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    if len(group) <= max_points:
        return group
    stride = math.ceil(len(group) / max_points)
    return group.iloc[::stride]


def plot_frame_delay_series(frame_delay: pd.DataFrame, output_path: Path) -> None:
    if frame_delay.empty:
        return

    sns.set_theme(style="darkgrid", context="talk")
    fig, ax = plt.subplots(figsize=(13, 7))

    for car_id, group in frame_delay.groupby("car_id"):
        subset = downsample(group.sort_values("time"))
        ax.plot(
            # subset["time"],
            np.arange(0, len(subset["value"])),  # x values
            subset["value"],
            label=f"Car {car_id}",
            linewidth=1.2,
            alpha=0.8,
        )

    # only doing for the first car to avoid clutter
    # if "car_id" in frame_delay.columns:
    #     first_car_id = frame_delay["car_id"].min()
    #     first_car_data = frame_delay[frame_delay["car_id"] == first_car_id]
    #     if not first_car_data.empty:
    #         # mean_delay = first_car_data["value"].mean()
    #         # ax.axhline(mean_delay, color="red", linestyle="--", label=f"Mean Delay Car {first_car_id}")
    #         subset = downsample(first_car_data.sort_values("time"))
    #         import numpy as np
    #         ax.plot(
    #             # subset["time"],

    #             # np.array([0, len(subset["value"]])]) * [0, 1],  # x=0 line
    #             # np.zeros(len(subset["value"])),  # y=0 line
    #             np.arange(0, len(subset["value"])),  # x values
    #             subset["value"],
    #             label=f"Car {first_car_id}",
    #             linestyle="None",
    #             marker="o",
    #             markersize=3,

    #             alpha=0.8,
    #         )
    #         print(f"car 0 no o fsamples ",len(subset["value"]))

    ax.set_title("VoIP Frame Delay per Car")
    ax.set_xlabel("Simulation Time [s]")
    ax.set_ylabel("Frame Delay [ms]")
    ax.legend(ncol=2, frameon=True, fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_frame_delay_summary(stats: pd.DataFrame, output_path: Path) -> None:
    if stats.empty:
        return

    sns.set_theme(style="whitegrid", context="talk")
    plot_df = stats.reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(plot_df["car_id"], plot_df["mean_delay"], color="#1f77b4", label="Mean")
    ax.errorbar(
        plot_df["car_id"],
        plot_df["mean_delay"],
        yerr=(plot_df["p95_delay"] - plot_df["mean_delay"]),
        fmt="none",
        ecolor="#ff7f0e",
        elinewidth=2,
        capsize=6,
        label="95th percentile",
    )
    ax.set_xlabel("Car ID")
    ax.set_ylabel("Delay [ms]")
    ax.set_title("VoIP Frame Delay summary (mean with 95th percentile)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_serving_cell_transitions(
    serving_cell: pd.DataFrame, output_path: Path
) -> None:
    if serving_cell.empty:
        return

    sns.set_theme(style="ticks", context="talk")
    unique_cars = sorted(serving_cell["car_id"].unique())
    rows = math.ceil(len(unique_cars) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(16, 3.0 * rows), sharex=True)
    axes = axes.flatten()

    for ax in axes[len(unique_cars) :]:
        ax.set_visible(False)

    for ax, car_id in zip(axes, unique_cars):
        group = serving_cell[serving_cell["car_id"] == car_id].sort_values("time")
        subset = downsample(group, max_points=1500)

        if subset.empty:
            ax.text(
                0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes
            )
        else:
            values = subset["value"].to_numpy()
            times = subset["time"].to_numpy()
            if len(subset) == 1 or np.allclose(values, values[0]):
                constant = float(values[0])
                xmin = float(times[0])
                xmax = float(times[-1])
                if math.isclose(xmax, xmin):
                    xmin -= 0.5
                    xmax += 0.5
                ax.hlines(
                    constant, xmin=xmin, xmax=xmax, colors="#1f77b4", linewidth=1.4
                )
                ax.scatter(times, values, s=28, color="#1f77b4", alpha=0.8, zorder=3)
                ax.text(
                    0.5,
                    0.1,
                    "No transitions",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize="small",
                )
                ax.set_xlim(xmin, xmax)
            else:
                ax.step(subset["time"], subset["value"], where="post", linewidth=1.3)

        ax.set_title(f"Car {car_id}")
        ax.set_ylabel("Serving Cell ID")
        ax.grid(True, which="both", axis="both")

    axes[0].set_xlabel("Simulation Time [s]")
    axes[1].set_xlabel("Simulation Time [s]")
    fig.suptitle("Serving Cell Selection over Time", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_throughput(
    throughput_generated: pd.DataFrame,
    throughput_received: pd.DataFrame,
    output_path: Path,
) -> None:
    if throughput_generated.empty and throughput_received.empty:
        return

    sns.set_theme(style="darkgrid", context="talk")
    fig, ax = plt.subplots(figsize=(13, 7))

    for label, df in (
        ("Generated", throughput_generated),
        ("Received", throughput_received),
    ):
        if df.empty:
            continue
        grouped = df.groupby("time")["value"].sum().reset_index()
        grouped = grouped.sort_values("time")
        subset = downsample(grouped, max_points=2500)
        ax.plot(subset["time"], subset["value"], label=f"Total {label}", linewidth=1.6)

    ax.set_title("Aggregate VoIP Throughput")
    ax.set_xlabel("Simulation Time [s]")
    ax.set_ylabel("Throughput [kbps]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def time_weighted_mean(group: pd.DataFrame) -> float:
    ordered = group.sort_values("time")
    times = ordered["time"].to_numpy()
    values = ordered["value"].to_numpy()
    if len(ordered) == 0:
        return float("nan")
    if len(ordered) == 1:
        return float(values[0])
    duration = times[-1] - times[0]
    if duration <= 0:
        return float(np.mean(values))
    integral = np.trapezoid(values, times)
    return float(integral / duration)


def plot_avg_throughput_per_car(
    throughput_generated: pd.DataFrame,
    throughput_received: pd.DataFrame,
    output_path: Path,
) -> None:
    frames: List[pd.DataFrame] = []
    for label, df in (
        ("Generated", throughput_generated),
        ("Received", throughput_received),
    ):
        if df.empty:
            continue
        rows: List[Dict[str, float]] = []
        for car_id, group in df.groupby("car_id"):
            avg = time_weighted_mean(group)
            if math.isnan(avg):
                continue
            rows.append({"car_id": car_id, "value": avg, "series": label})
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return

    plot_df = pd.concat(frames, ignore_index=True)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=plot_df, x="car_id", y="value", hue="series", ax=ax)
    ax.set_xlabel("Car ID")
    ax.set_ylabel("Average Throughput")
    ax.set_title("Time-weighted Average VoIP Throughput per Car")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def write_summary_csv(stats: pd.DataFrame, output_path: Path) -> None:
    if stats.empty:
        return
    stats.sort_index().to_csv(output_path, float_format="%.6f")


def run_analysis(input_path: Path, output_dir: Path, max_clients: int) -> None:
    header, total_cols = load_header(input_path)
    metric_map = assemble_metric_map(header, max_clients=max_clients)
    data, _, _ = build_dataframe(input_path, total_cols, metric_map)

    if data.empty:
        raise RuntimeError(
            "No metrics were loaded; check that the CSV contains the expected signals."
        )

    frame_delay = build_long_form(data, "frame_delay", metric_map["frame_delay"])
    serving_cell = build_long_form(data, "serving_cell", metric_map["serving_cell"])
    generated_tp = build_long_form(
        data, "generated_throughput", metric_map["generated_throughput"]
    )
    received_tp = build_long_form(
        data, "received_throughput", metric_map["received_throughput"]
    )

    if not generated_tp.empty:
        generated_tp = generated_tp.copy()
        generated_tp["value"] = generated_tp["value"] * GENERATED_THROUGHPUT_SCALE

    summary = compute_frame_delay_stats(frame_delay, serving_cell)

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_frame_delay_series(frame_delay, output_dir / "voip_frame_delay_timeseries.png")
    plot_frame_delay_summary(summary, output_dir / "voip_frame_delay_summary.png")
    plot_serving_cell_transitions(
        serving_cell, output_dir / "serving_cell_transitions.png"
    )
    plot_throughput(generated_tp, received_tp, output_dir / "voip_throughput_total.png")
    plot_avg_throughput_per_car(
        generated_tp,
        received_tp,
        output_dir / "voip_throughput_per_car.png",
    )
    write_summary_csv(summary, output_dir / "voip_frame_delay_summary.csv")

    if not summary.empty:
        print("Frame delay summary (first rows):")
        print(summary.head())
    else:
        print("No frame delay data detected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse OMNeT++ VoIP latency exports."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("omnet-data/latency_with_10cars2RSU_30.09.2025.csv"),
        help="Path to the exported CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store plots and summary tables (defaults to <input_dir>/analysis_output).",
    )
    parser.add_argument(
        "--max-clients",
        type=int,
        default=MAX_DEFAULT_CLIENTS,
        help="Number of client vehicles to inspect (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_dir: Path
    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = input_path.parent / "analysis_output"

    run_analysis(input_path, output_dir, max_clients=args.max_clients)
    print(f"Analysis artefacts written to: {output_dir}")


if __name__ == "__main__":
    main()
