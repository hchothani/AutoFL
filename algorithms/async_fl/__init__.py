"""Async FL module for asynchronous federated learning.

This module provides components for running asynchronous federated learning
with Flower, where clients can train and update the global model independently
without waiting for synchronization rounds.

Components:
    - AsynchronousStrategy: Async aggregation strategies (FedAsync, AsyncFedED)
    - AsyncServer: Server with concurrent client execution
    - AsyncClientManager: Client manager with free/busy state tracking
    - AsyncHistory: Timestamp-based metrics tracking
    - SimulatedAsyncClient: Simulated client for async FL simulations
"""

from .async_strategy import AsynchronousStrategy
# from .async_server import AsyncServer, fit_clients, evaluate_clients
# from .async_client_manager import AsyncClientManager
from .async_history import AsyncHistory
# from .simulated_client import (
#    SimulatedAsyncClient,
#    SimulatedClientConfig,
#    create_simulated_clients,
)

__all__ = [
    "AsynchronousStrategy",
#    "AsyncServer",
#    "AsyncClientManager",
    "AsyncHistory",
#    "fit_clients",
#    "evaluate_clients",
#    "SimulatedAsyncClient",
#    "SimulatedClientConfig",
#    "create_simulated_clients",
]
