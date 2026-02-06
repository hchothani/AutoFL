from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Tuple


class BenchmarkAdapter:
    """Utility wrapper that normalizes access to Avalanche benchmark streams."""

    _STREAM_ATTRS = {
        "train": ("train_stream", "train_datasets_stream"),
        "test": (
            "test_stream",
            "test_datasets_stream",
            "validation_stream",
            "valid_stream",
            "eval_stream",
            "eval_datasets_stream",
        ),
    }

    def __init__(self, benchmark: Any) -> None:
        self._benchmark = benchmark
        self.train_attr, self.train_stream = self._resolve_stream("train")
        self.test_attr, self.test_stream = self._resolve_stream("test")
        self.train_stream_name = self._derive_name(self.train_stream, self.train_attr)
        self.test_stream_name = self._derive_name(self.test_stream, self.test_attr)
        self.num_train_experiences = self._stream_length(self.train_stream)
        self.num_test_experiences = self._stream_length(self.test_stream)
        if self.num_train_experiences == 0:
            raise ValueError("Benchmark does not provide any training experiences")
        if self.num_test_experiences == 0:
            raise ValueError("Benchmark does not provide any evaluation experiences")

    def _resolve_stream(self, stream_type: str) -> Tuple[str, Any]:
        for attr in self._STREAM_ATTRS.get(stream_type, ()):  # type: ignore[arg-type]
            stream = getattr(self._benchmark, attr, None)
            if stream is not None:
                return attr, stream
        raise AttributeError(
            f"Benchmark {type(self._benchmark)} does not provide a '{stream_type}' stream"
        )

    def effective_experience_count(
        self,
        configured_lengths: Iterable[Any] | None,
        configured_default: int,
    ) -> int:
        configured_lengths = tuple(configured_lengths or ())
        configured_count = len(configured_lengths)
        if configured_count and configured_count != self.num_train_experiences:
            warnings.warn(
                (
                    "Configured experience count (%s) does not match benchmark stream length (%s); "
                    "falling back to benchmark length."
                )
                % (configured_count, self.num_train_experiences),
                RuntimeWarning,
            )
            return self.num_train_experiences
        if configured_count:
            return configured_count
        if self.num_train_experiences:
            return self.num_train_experiences
        return configured_default

    def training_experience(self, index: int):
        return self.train_stream[index % self.num_train_experiences]

    def extract_stream_metric(
        self,
        metrics: Dict[str, Any],
        metric_prefix: str,
        stream_name: str | None = None,
        phase: str = "eval_phase",
    ) -> float:
        target_stream = stream_name or self.test_stream_name
        primary_key = f"{metric_prefix}/{phase}/{target_stream}"
        if primary_key in metrics:
            return float(metrics[primary_key])
        for key, value in metrics.items():
            if key.startswith(metric_prefix) and f"/{phase}/" in key and key.endswith(target_stream):
                return float(value)
        for key, value in metrics.items():
            if key.startswith(metric_prefix) and f"/{phase}/" in key:
                return float(value)
        for key, value in metrics.items():
            if key.startswith(metric_prefix):
                return float(value)
        return 0.0

    @staticmethod
    def _derive_name(stream: Any, attr_name: str) -> str:
        stream_name = getattr(stream, "name", None)
        if stream_name:
            return stream_name
        if attr_name:
            return attr_name.replace("_stream", "")
        return stream.__class__.__name__

    @staticmethod
    def _stream_length(stream: Any) -> int:
        try:
            return len(stream)  # type: ignore[arg-type]
        except TypeError:
            # Fall back to materializing the stream once.
            cached = tuple(stream)
            return len(cached)