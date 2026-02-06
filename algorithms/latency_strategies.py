"""Latency-aware Flower server strategies."""

from __future__ import annotations

import math
import time
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes

import wandb

from utils.latency_simulator import get_runtime_recorder


class LatencyAwareFedAvg(FedAvg):
    """FedAvg variant that skips dropped clients and logs aggregation runtime."""

    def __init__(self, *args, latency_cfg: Optional[Mapping[str, object]] = None, **kwargs):
        self._latency_cfg = dict(latency_cfg) if latency_cfg is not None else {}
        self._latency_sampling_mode = str(self._latency_cfg.get("sampling_mode", "mean")).lower()
        self._skip_if_slow_aggregation = bool(self._latency_cfg.get("skip_if_slow_aggregation", False))
        self._skip_if_slow_margin = float(self._latency_cfg.get("skip_if_slow_margin", 1.0))
        self._sleep_log_enabled = bool(
            self._latency_cfg.get("sleep_log", self._latency_cfg.get("sleep", False))
        )
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: Sequence[Tuple[ClientProxy, FitRes]],
        failures: Sequence[BaseException],
    ):
        dropped_clients: List[int] = []
        dropped_entries = []
        filtered_results = []
        expected_times: List[float] = []
        for client_proxy, fit_res in results:
            metrics = dict(fit_res.metrics)
            expected_time = float(metrics.get("latency/expected_network_time_s", 0.0))
            if metrics.get("latency/dropped", False):
                client_id = int(metrics.get("pid", -1))
                dropped_clients.append(client_id)
                dropped_entries.append((fit_res.num_examples, metrics))
                continue
            if expected_time > 0:
                expected_times.append(expected_time)
            filtered_results.append((client_proxy, fit_res))

        start_time = time.time()
        aggregated = super().aggregate_fit(server_round, filtered_results, failures)
        duration_wall_clock = time.time() - start_time

        latency_component = 0.0
        if self._sleep_log_enabled and expected_times:
            finite_expected = [value for value in expected_times if math.isfinite(value) and value >= 0.0]
            if finite_expected:
                latency_component = float(np.max(finite_expected))

        duration_reported = duration_wall_clock + latency_component

        expected_mean = float(np.mean(expected_times)) if expected_times else float("nan")
        expected_max = float(np.max(expected_times)) if expected_times else float("nan")
        aggregation_threshold = (
            expected_max * max(self._skip_if_slow_margin, 1e-6)
            if expected_times and expected_max > 0
            else float("nan")
        )
        aggregation_delta = duration_reported - expected_mean if expected_times else float("nan")
        skipped_due_to_latency = False

        if (
            self._skip_if_slow_aggregation
            and self._latency_sampling_mode == "mean"
            and expected_times
            and expected_max > 0
        ):
            threshold = aggregation_threshold
            if duration_reported > threshold:
                skipped_due_to_latency = True
                print(
                    f"[LatencyAwareFedAvg] Skipping aggregation for round {server_round}: "
                    f"{duration_reported:.3f}s > threshold {threshold:.3f}s"
                )
                aggregated = None

        recorder = get_runtime_recorder()
        if recorder is not None:
            recorder.log_server_round(
                round_id=server_round,
                aggregation_time_s=duration_reported,
                total_results=len(results),
                accepted_results=len(filtered_results),
                dropped_clients=dropped_clients,
                expected_mean_network_time_s=expected_mean,
                expected_max_network_time_s=expected_max,
                skipped_due_to_latency=skipped_due_to_latency,
                aggregation_threshold_s=aggregation_threshold,
                aggregation_minus_expected_mean_s=aggregation_delta,
                aggregation_wall_clock_s=duration_wall_clock,
                aggregation_latency_component_s=latency_component,
            )
            if dropped_entries:
                recorder.log_client_round(
                    server_round,
                    dropped_entries,
                    dropped_clients=dropped_clients,
                )

        wandb.log(
            {
                "server/aggregation_time_s": duration_reported,
                "server/aggregation_wall_clock_s": duration_wall_clock,
                "server/aggregation_latency_component_s": latency_component,
                "server/total_results": len(results), # KIND OF Useless (but keeping)
                "server/accepted_results": len(filtered_results),
                "server/dropped_clients": len(dropped_clients),
                "server/expected_mean_network_time_s": expected_mean,
                "server/expected_max_network_time_s": expected_max,
                "server/aggregation_threshold_s": aggregation_threshold,
                "server/aggregation_vs_expected_delta_s": aggregation_delta,
                "server/skipped_due_to_latency": skipped_due_to_latency,
            },
            step=server_round,
        )
        return aggregated


STRATEGY_REGISTRY = {
    "latency_aware_fedavg": LatencyAwareFedAvg,
    "latencyawarefedavg": LatencyAwareFedAvg,
    "fedavg": LatencyAwareFedAvg,
}
