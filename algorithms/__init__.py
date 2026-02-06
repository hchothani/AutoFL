"""Strategy registry"""

from .latency_strategies import LatencyAwareFedAvg, STRATEGY_REGISTRY

__all__ = [
    "LatencyAwareFedAvg",
    "STRATEGY_REGISTRY",
]
