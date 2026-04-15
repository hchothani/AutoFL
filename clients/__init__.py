"""Client utilities for Flower-based continual learning."""

from .flower_client import FlowerClient, initialize_partition_strategies

__all__ = ["FlowerClient", "initialize_partition_strategies"]
