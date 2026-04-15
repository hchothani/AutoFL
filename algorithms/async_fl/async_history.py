"""Async History module for Flower async FL.

A wrapper around the Flower History class that offers centralized and distributed
metrics per timestamp instead of per round. It also groups distributed_fit metrics
per client instead of per round.

The latest Flower (1.25.0+) History class uses server_round (int) as the key,
but for asynchronous FL we need to track timestamps. This class extends History
to support both paradigms.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from flwr.common.typing import Scalar
from flwr.server.history import History


class AsyncHistory(History):
    """Extended History class for asynchronous federated learning.

    This class extends the base Flower History class to support timestamp-based
    metrics tracking required for asynchronous federated learning scenarios.

    Attributes:
        metrics_distributed_fit_async: Dict mapping metric names to dict of
            client_id -> list of (timestamp, value) tuples
        metrics_centralized_async: Dict mapping metric names to list of
            (timestamp, value) tuples
        losses_centralized_async: List of (timestamp, loss) tuples
    """

    def __init__(self) -> None:
        """Initialize AsyncHistory with async-specific tracking structures."""
        super().__init__()
        # Async-specific tracking structures using timestamps
        self.metrics_distributed_fit_async: Dict[
            str, Dict[str, List[Tuple[float, Scalar]]]
        ] = {}
        self.metrics_centralized_async: Dict[str, List[Tuple[float, Scalar]]] = {}
        self.losses_centralized_async: List[Tuple[float, float]] = []

    def add_metrics_distributed_fit_async(
        self, client_id: str, metrics: Dict[str, Scalar], timestamp: float
    ) -> None:
        """Add metrics entries from distributed fit, indexed by client and timestamp.

        Args:
            client_id: The client identifier
            metrics: Dictionary of metric name to value
            timestamp: The timestamp when these metrics were recorded
        """
        for key in metrics:
            if key not in self.metrics_distributed_fit_async:
                self.metrics_distributed_fit_async[key] = {}
            if client_id not in self.metrics_distributed_fit_async[key]:
                self.metrics_distributed_fit_async[key][client_id] = []
            self.metrics_distributed_fit_async[key][client_id].append(
                (timestamp, metrics[key])
            )

    def add_metrics_centralized_async(
        self, metrics: Dict[str, Scalar], timestamp: float
    ) -> None:
        """Add metrics entries from centralized evaluation with timestamp.

        Args:
            metrics: Dictionary of metric name to value
            timestamp: The timestamp when these metrics were recorded
        """
        for metric in metrics:
            if metric not in self.metrics_centralized_async:
                self.metrics_centralized_async[metric] = []
            self.metrics_centralized_async[metric].append((timestamp, metrics[metric]))

    def add_loss_centralized_async(self, timestamp: float, loss: float) -> None:
        """Add loss entry from centralized evaluation with timestamp.

        Args:
            timestamp: The timestamp when this loss was recorded
            loss: The loss value
        """
        self.losses_centralized_async.append((timestamp, loss))

    def get_async_metrics_summary(self) -> Dict[str, any]:
        """Get a summary of async metrics for logging.

        Returns:
            Dictionary with summary statistics of async metrics
        """
        summary = {
            "num_centralized_evaluations": len(self.losses_centralized_async),
            "num_distributed_fit_metrics": len(self.metrics_distributed_fit_async),
        }

        if self.losses_centralized_async:
            summary["final_centralized_loss"] = self.losses_centralized_async[-1][1]
            summary["total_training_time"] = (
                self.losses_centralized_async[-1][0]
                - self.losses_centralized_async[0][0]
            )

        return summary
