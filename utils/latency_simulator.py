"""Latency simulation utilities using OMNeT++ exports."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig


@dataclass(frozen=True)
class MetricLocation:
    time_idx: int
    value_idx: int
    raw_label: str

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

    rename_map = {f"col_{idx}": index_to_name[idx] for idx in sorted_usecols if idx in index_to_name}
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


@dataclass
class ClientLatencySample:
    """Latency sample returned to the simulator caller."""

    base_delay_s: float
    download_time_s: float
    upload_time_s: float
    total_network_time_s: float
    threshold_s: float
    exceeded_threshold: bool


@dataclass
class ClientLatencyStats:
    """Aggregate statistics and traces per client."""

    mean_delay_s: float
    std_delay_s: float
    delay_trace: np.ndarray
    generated_trace: np.ndarray
    received_trace: np.ndarray
    generated_mean_kbps: float
    received_mean_kbps: float
    delay_chunk_means: np.ndarray


class LatencySimulator:
    """Sample OMNeT++ derived latency/throughput metrics for AutoFL clients."""

    def __init__(self, cfg: DictConfig):
        latency_cfg = cfg.get("latency", {})
        self.enabled: bool = bool(latency_cfg.get("enabled", False))
        self.sampling_mode: str = str(latency_cfg.get("sampling_mode", "mean")).lower()
        self.scaling_factor: float = float(latency_cfg.get("scaling_factor", 1.0))
        self.threshold_multiplier: float = float(latency_cfg.get("threshold_multiplier", 1.0))
        self.drop_behavior: str = str(latency_cfg.get("drop_behavior", "skip")).lower()
        configured_sleep = bool(latency_cfg.get("sleep", False))
        self.sleep_log: bool = bool(latency_cfg.get("sleep_log", configured_sleep))
        self.random_seed: int = int(latency_cfg.get("random_seed", 42))
        self.max_clients: int = int(latency_cfg.get("max_clients", 10))
        self.throughput_floor_kbps: float = float(latency_cfg.get("throughput_floor_kbps", 10.0))
        self.upload_multiplier: float = float(latency_cfg.get("upload_multiplier", 1.0))
        self.download_multiplier: float = float(latency_cfg.get("download_multiplier", 1.0))
        csv_path = latency_cfg.get("csv_path", "")
        self.csv_path = Path(csv_path) if csv_path else None
        server_cfg = cfg.get("server", {}) or {}
        self.total_rounds = int(server_cfg.get("num_rounds", 1)) if server_cfg is not None else 1
        self.skip_if_slow_aggregation: bool = bool(latency_cfg.get("skip_if_slow_aggregation", False))
        self.skip_if_slow_margin: float = float(latency_cfg.get("skip_if_slow_margin", 1.0))
        self.log_round_time_variance: bool = bool(latency_cfg.get("log_round_time_variance", False))

        self._rng = np.random.default_rng(self.random_seed)
        self._stats: Dict[int, ClientLatencyStats] = {}
        if self.enabled:
            if not self.csv_path:
                raise ValueError("latency.enabled is True but latency.csv_path is not set")
            if not self.csv_path.exists():
                raise FileNotFoundError(f"Latency CSV not found: {self.csv_path}")
            self._load_metrics()

    def _load_metrics(self) -> None:
        header, total_cols = load_header(self.csv_path)
        metric_map = assemble_metric_map(header, max_clients=self.max_clients)
        data, _, _ = build_dataframe(self.csv_path, total_cols, metric_map)
        if data.empty:
            raise RuntimeError("Failed to extract latency metrics from CSV")

        frame_df = build_long_form(data, "frame_delay", metric_map["frame_delay"])
        generated_df = build_long_form(data, "generated_throughput", metric_map["generated_throughput"])
        received_df = build_long_form(data, "received_throughput", metric_map["received_throughput"])

        # Convert generated throughput from kBytes/s to kbits/s if not already scaled.
        # analyze_latency multiplies generated throughput by scale when plotting; reproduce here.
        if not generated_df.empty:
            generated_df = generated_df.copy()
            generated_df["value"] = generated_df["value"] * 8.0

        for client_id in range(self.max_clients):
            delay_trace = self._extract_trace(frame_df, client_id)
            if delay_trace is None:
                continue
            generated_trace = self._extract_trace(generated_df, client_id, allow_empty=True)
            received_trace = self._extract_trace(received_df, client_id, allow_empty=True)

            mean_delay_s = float(np.mean(delay_trace)) if delay_trace.size else 0.0
            std_delay_s = float(np.std(delay_trace)) if delay_trace.size else 0.0

            gen_mean = self._time_weighted_average(generated_df, client_id)
            rec_mean = self._time_weighted_average(received_df, client_id)
            delay_chunk_means = self._compute_chunk_means(delay_trace)

            stats = ClientLatencyStats(
                mean_delay_s=mean_delay_s,
                std_delay_s=std_delay_s,
                delay_trace=delay_trace,
                generated_trace=generated_trace if generated_trace is not None else np.array([]),
                received_trace=received_trace if received_trace is not None else np.array([]),
                generated_mean_kbps=gen_mean,
                received_mean_kbps=rec_mean,
                delay_chunk_means=delay_chunk_means,
            )
            self._stats[client_id] = stats

    @staticmethod
    def _extract_trace(df: pd.DataFrame, client_id: int, allow_empty: bool = False) -> Optional[np.ndarray]:
        if df.empty:
            return np.array([]) if allow_empty else None
        subset = df[df["car_id"] == client_id]
        if subset.empty:
            return np.array([]) if allow_empty else None
        return subset["value"].to_numpy(copy=True)

    @staticmethod
    def _time_weighted_average(df: pd.DataFrame, client_id: int) -> float:
        if df.empty:
            return float("nan")
        subset = df[df["car_id"] == client_id]
        if subset.empty:
            return float("nan")
        return float(time_weighted_mean(subset))

    def _compute_chunk_means(self, trace: np.ndarray) -> np.ndarray:
        if trace.size == 0 or self.total_rounds <= 0:
            return np.array([])
        chunks = np.array_split(trace, self.total_rounds)
        fallback = float(np.mean(trace)) if trace.size else 0.0
        means: List[float] = []
        for chunk in chunks:
            if chunk.size == 0:
                means.append(fallback)
            else:
                means.append(float(np.mean(chunk)))
        return np.array(means, dtype=float)

    def has_client(self, client_id: int) -> bool:
        return client_id in self._stats

    def sample(self, client_id: int, round_idx: int, payload_bytes: int) -> ClientLatencySample:
        if not self.enabled or client_id not in self._stats:
            return ClientLatencySample(
                base_delay_s=0.0,
                download_time_s=0.0,
                upload_time_s=0.0,
                total_network_time_s=0.0,
                threshold_s=float("inf"),
                exceeded_threshold=False,
            )

        stats = self._stats[client_id]
        base_delay_s = self._sample_delay(stats, round_idx)
        download_time_s = self._compute_transfer_time(
            payload_bytes,
            rate_kbps=self._sample_throughput(stats.received_trace, stats.received_mean_kbps, round_idx),
            multiplier=self.download_multiplier,
        )
        upload_time_s = self._compute_transfer_time(
            payload_bytes,
            rate_kbps=self._sample_throughput(stats.generated_trace, stats.generated_mean_kbps, round_idx),
            multiplier=self.upload_multiplier,
        )

        total_network_time_s = self.scaling_factor * (base_delay_s + download_time_s + upload_time_s)

        threshold_s = self.scaling_factor * (
            stats.mean_delay_s * self.threshold_multiplier + (self.threshold_multiplier * self._safe_value(stats.std_delay_s))
        )
        # TODO: modified this here temperorily for even mean to be higher
        exceeded = total_network_time_s > threshold_s if math.isfinite(threshold_s) else False

        sample = ClientLatencySample(
            base_delay_s=base_delay_s * self.scaling_factor,
            download_time_s=download_time_s * self.scaling_factor,
            upload_time_s=upload_time_s * self.scaling_factor,
            total_network_time_s=total_network_time_s,
            threshold_s=threshold_s,
            exceeded_threshold=exceeded,
        )

        return sample

    def should_remove_permanently(self) -> bool:
        return self.drop_behavior == "remove"

    def _sample_delay(self, stats: ClientLatencyStats, round_idx: int) -> float:
        # Delay trace is in seconds already.
        if stats.delay_trace.size == 0:
            return self._safe_value(stats.mean_delay_s)
        if self.sampling_mode == "mean":
            return self._safe_value(stats.mean_delay_s)
        if self.sampling_mode == "chunk":
            if stats.delay_chunk_means.size == 0:
                return self._safe_value(stats.mean_delay_s)
            index = max(0, min(round_idx - 1, stats.delay_chunk_means.size - 1))
            return float(stats.delay_chunk_means[index])
        if self.sampling_mode == "trace":
            index = round_idx % stats.delay_trace.size
            return float(stats.delay_trace[index])
        if self.sampling_mode == "random":
            return float(self._rng.choice(stats.delay_trace)) if stats.delay_trace.size else self._safe_value(stats.mean_delay_s)
        raise ValueError(f"Unsupported latency.sampling_mode: {self.sampling_mode}")

    def _sample_throughput(
        self,
        trace: np.ndarray,
        mean_kbps: float,
        round_idx: int,
    ) -> float:
        candidate = float(mean_kbps) if not math.isnan(mean_kbps) else 0.0
        if self.sampling_mode in {"mean", "chunk"} or trace.size == 0:
            return max(candidate, self.throughput_floor_kbps)
        if self.sampling_mode == "trace":
            index = round_idx % trace.size
            candidate = float(trace[index])
        elif self.sampling_mode == "random":
            candidate = float(self._rng.choice(trace))
        return max(candidate, self.throughput_floor_kbps)

    def _compute_transfer_time(self, payload_bytes: int, rate_kbps: float, multiplier: float) -> float:
        if payload_bytes <= 0:
            return 0.0
        bits = payload_bytes * 8.0
        rate_bps = max(rate_kbps * 1000.0, self.throughput_floor_kbps * 1000.0)
        return (bits / rate_bps) * multiplier

    @staticmethod
    def _safe_value(value: float) -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value

    @property
    def sleep_log_enabled(self) -> bool:
        return self.sleep_log

    def sleep_pre_training(self, sample: ClientLatencySample) -> float:
        if not self.sleep_log or sample is None:
            return 0.0
        total = max(sample.base_delay_s + sample.download_time_s, 0.0)
        return total

    def sleep_post_training(self, sample: ClientLatencySample) -> float:
        if not self.sleep_log or sample is None:
            return 0.0
        return max(sample.upload_time_s, 0.0)


# ----------------------------------------------------------------------
# Runtime metric capture helpers
# ----------------------------------------------------------------------

@dataclass
class ClientRoundRecord:
    round_id: int
    client_id: int
    num_examples: int
    metrics: Dict[str, float]
    dropped: bool


@dataclass
class ServerRoundRecord:
    round_id: int
    aggregation_time_s: float
    aggregation_wall_clock_s: float
    aggregation_latency_component_s: float
    total_results: int
    accepted_results: int
    dropped_clients: List[int]
    expected_mean_network_time_s: float
    expected_max_network_time_s: float
    skipped_due_to_latency: bool
    aggregation_threshold_s: float
    aggregation_minus_expected_mean_s: float


class RuntimeMetricsRecorder:
    """Collect latency/timing metrics for persistence and plotting."""

    def __init__(self, output_dir: Path, save_client: bool = True, save_server: bool = True) -> None:
        self.output_dir = output_dir
        self.save_client = save_client
        self.save_server = save_server
        self._client_records: List[ClientRoundRecord] = []
        self._server_records: List[ServerRoundRecord] = []
        self._aggregate_records: List[Dict[str, object]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_client_round(
        self,
        round_id: int,
        entries: List[Tuple[int, Dict[str, float]]],
        dropped_clients: Optional[List[int]] = None,
    ) -> None:
        if not self.save_client:
            return
        dropped_clients = dropped_clients or []
        dropped_set = set(dropped_clients)
        for num_examples, metrics in entries:
            client_id = int(metrics.get("pid", -1))
            record = ClientRoundRecord(
                round_id=round_id,
                client_id=client_id,
                num_examples=num_examples,
                metrics=metrics,
                dropped=bool(metrics.get("latency/dropped", False) or client_id in dropped_set),
            )
            self._client_records.append(record)
        # Record explicit dropouts (clients filtered before metrics aggregation)
        for client_id in dropped_clients:
            record = ClientRoundRecord(
                round_id=round_id,
                client_id=client_id,
                num_examples=0,
                metrics={"latency/dropped": True},
                dropped=True,
            )
            self._client_records.append(record)

    def log_server_round(
        self,
        round_id: int,
        aggregation_time_s: float,
        total_results: int,
        accepted_results: int,
        dropped_clients: Optional[List[int]] = None,
        expected_mean_network_time_s: float = float("nan"),
        expected_max_network_time_s: float = float("nan"),
        skipped_due_to_latency: bool = False,
        aggregation_threshold_s: float = float("nan"),
        aggregation_minus_expected_mean_s: float = float("nan"),
        aggregation_wall_clock_s: float = float("nan"),
        aggregation_latency_component_s: float = float("nan"),
    ) -> None:
        if not self.save_server:
            return
        record = ServerRoundRecord(
            round_id=round_id,
            aggregation_time_s=aggregation_time_s,
            aggregation_wall_clock_s=aggregation_wall_clock_s,
            aggregation_latency_component_s=aggregation_latency_component_s,
            total_results=total_results,
            accepted_results=accepted_results,
            dropped_clients=dropped_clients or [],
            expected_mean_network_time_s=expected_mean_network_time_s,
            expected_max_network_time_s=expected_max_network_time_s,
            skipped_due_to_latency=skipped_due_to_latency,
            aggregation_threshold_s=aggregation_threshold_s,
            aggregation_minus_expected_mean_s=aggregation_minus_expected_mean_s,
        )
        self._server_records.append(record)

    def log_aggregate_metrics(self, round_id: int, phase: str, metrics: Mapping[str, object]) -> None:
        row: Dict[str, object] = {"round": round_id, "phase": phase}
        for key, value in metrics.items():
            if isinstance(value, (dict, list, tuple)):
                row[key] = json.dumps(value)
            else:
                row[key] = value
        self._aggregate_records.append(row)

    def flush(self) -> None:
        if self.save_client:
            client_path = self.output_dir / "client_round_metrics.csv"
            if self._client_records:
                df = pd.DataFrame(
                    [
                        {
                            **record.metrics,
                            "round": record.round_id,
                            "client_id": record.client_id,
                            "num_examples": record.num_examples,
                            "dropped": record.dropped,
                        }
                        for record in self._client_records
                    ]
                )
                df.sort_values(["round", "client_id"], inplace=True)
                df.to_csv(client_path, index=False)
        if self.save_server:
            server_path = self.output_dir / "server_round_metrics.csv"
            if self._server_records:
                df = pd.DataFrame(
                    [
                        {
                            "round": record.round_id,
                            "aggregation_time_s": record.aggregation_time_s,
                            "aggregation_wall_clock_s": record.aggregation_wall_clock_s,
                            "aggregation_latency_component_s": record.aggregation_latency_component_s,
                            "total_results": record.total_results,
                            "accepted_results": record.accepted_results,
                            "dropped_clients": ",".join(map(str, record.dropped_clients)),
                            "expected_mean_network_time_s": record.expected_mean_network_time_s,
                            "expected_max_network_time_s": record.expected_max_network_time_s,
                            "skipped_due_to_latency": record.skipped_due_to_latency,
                            "aggregation_threshold_s": record.aggregation_threshold_s,
                            "aggregation_minus_expected_mean_s": record.aggregation_minus_expected_mean_s,
                        }
                        for record in self._server_records
                    ]
                )
                df.sort_values("round", inplace=True)
                df.to_csv(server_path, index=False)
        if self._aggregate_records:
            aggregate_path = self.output_dir / "aggregate_metrics.csv"
            df = pd.DataFrame(self._aggregate_records)
            df.sort_values(["phase", "round"], inplace=True)
            df.to_csv(aggregate_path, index=False)


_GLOBAL_RECORDER: Optional[RuntimeMetricsRecorder] = None


def init_runtime_recorder(cfg: DictConfig) -> RuntimeMetricsRecorder:
    global _GLOBAL_RECORDER
    logging_cfg = cfg.get("logging", {})
    output_root = Path(logging_cfg.get("output_root", "outputs"))
    output_dir = Path(cfg.get("logging", {}).get("run_output_dir", output_root))
    save_client = bool(logging_cfg.get("save_client_metrics", True))
    save_server = bool(logging_cfg.get("save_server_metrics", True))
    recorder = RuntimeMetricsRecorder(output_dir=output_dir, save_client=save_client, save_server=save_server)
    _GLOBAL_RECORDER = recorder
    return recorder


def get_runtime_recorder() -> Optional[RuntimeMetricsRecorder]:
    return _GLOBAL_RECORDER


def flush_runtime_recorder() -> None:
    recorder = get_runtime_recorder()
    if recorder is not None:
        recorder.flush()
