"""Simulated async client for async federated learning.

This module provides a simulated client that can be used with AsyncServer
for running async FL simulations without real network communication.
"""

from __future__ import annotations

import time
import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from flwr.common import (
    Code,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from algorithms.plora import inject_lora, get_lora_parameters, set_lora_parameters


@dataclass
class SimulatedClientConfig:
    """Configuration for simulated async client."""

    client_id: str
    model_fn: callable  # Function to create model
    train_loader: DataLoader
    test_loader: DataLoader
    device: torch.device
    local_epochs: int = 2
    learning_rate: float = 0.01
    simulate_delay: bool = True
    min_delay: float = 0.5
    max_delay: float = 3.0
    aggregation_strategy: str = "fedasync"
    strategy_params: Dict[str, Any] | None = None


class SimulatedAsyncClient(ClientProxy):
    """Simulated client for async FL that trains locally.

    This client simulates local training with configurable delays
    to mimic real-world async behavior.
    """

    def __init__(self, config: SimulatedClientConfig):
        super().__init__(config.client_id)
        self.config = config
        self.aggregation_strategy = str(config.aggregation_strategy).lower()
        self.strategy_params = config.strategy_params or {}

        if self.aggregation_strategy == "plora":
            init_seed = self.strategy_params.get("model_init_seed")
            if init_seed is not None:
                torch.manual_seed(int(init_seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(init_seed))

        model = config.model_fn()
        if self.aggregation_strategy == "plora":
            model = inject_lora(
                model,
                rank=int(self.strategy_params.get("rank", 4)),
                alpha=float(self.strategy_params.get("alpha", 1.0)),
            )
        self.model = model.to(config.device)
        self.train_loader = config.train_loader
        self.test_loader = config.test_loader
        self.device = config.device
        self.local_epochs = config.local_epochs
        self.learning_rate = config.learning_rate
        self._num_examples = len(config.train_loader.dataset)

    def _export_parameters(self) -> List[np.ndarray]:
        if self.aggregation_strategy == "plora":
            return get_lora_parameters(self.model)
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def _import_parameters(self, parameters: List[np.ndarray]) -> None:
        if self.aggregation_strategy == "plora":
            set_lora_parameters(self.model, parameters)
            return
        state_dict = self.model.state_dict()
        for key, param in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(param).to(self.device)
        self.model.load_state_dict(state_dict)

    def _trainable_parameters(self):
        if self.aggregation_strategy == "plora":
            return [
                p
                for name, p in self.model.named_parameters()
                if "lora_" in name and p.requires_grad
            ]
        return self.model.parameters()

    def get_parameters(
        self, ins: GetParametersIns, timeout: Optional[float] = None
    ) -> GetParametersRes:
        """Get current model parameters."""
        del ins, timeout
        params = self._export_parameters()
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(params),
        )

    def get_properties(self, ins, timeout: Optional[float] = None):
        """Get client properties."""
        from flwr.common import GetPropertiesRes

        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={
                "client_id": self.cid,
                "num_examples": self._num_examples,
            },
        )

    def fit(self, ins: FitIns, timeout: Optional[float] = None) -> FitRes:
        """Train the model on local data."""
        start_time = time.time()

        # Simulate network delay (download)
        if self.config.simulate_delay:
            download_delay = random.uniform(
                self.config.min_delay / 2, self.config.max_delay / 2
            )
            time.sleep(download_delay)

        # Set parameters from server
        params = parameters_to_ndarrays(ins.parameters)
        self._import_parameters(params)

        # Train locally
        self.model.train()
        trainable_params = self._trainable_parameters()
        optimizer = torch.optim.SGD(trainable_params, lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0
        for epoch in range(self.local_epochs):
            for batch in self.train_loader:
                if isinstance(batch, dict):
                    images = batch.get("img", batch.get("x")).to(self.device)
                    labels = batch.get("label", batch.get("y")).to(self.device)
                elif isinstance(batch, (tuple, list)):
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Simulate network delay (upload)
        if self.config.simulate_delay:
            upload_delay = random.uniform(
                self.config.min_delay / 2, self.config.max_delay / 2
            )
            time.sleep(upload_delay)

        # Get updated parameters
        new_params = self._export_parameters()

        elapsed = time.time() - start_time

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(new_params),
            num_examples=self._num_examples,
            metrics={
                "loss": avg_loss,
                "training_time": elapsed,
                "start_timestamp": ins.config.get("start_timestamp", start_time),
                "client_id": self.cid,
            },
        )

    def evaluate(
        self, ins: EvaluateIns, timeout: Optional[float] = None
    ) -> EvaluateRes:
        """Evaluate the model on local test data."""
        # Set parameters
        params = parameters_to_ndarrays(ins.parameters)
        self._import_parameters(params)

        # Evaluate
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, dict):
                    images = batch.get("img", batch.get("x")).to(self.device)
                    labels = batch.get("label", batch.get("y")).to(self.device)
                elif isinstance(batch, (tuple, list)):
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=avg_loss,
            num_examples=total,
            metrics={"accuracy": accuracy},
        )

    def reconnect(self, ins, timeout=None):
        """Handle reconnection request."""
        from flwr.common import DisconnectRes

        return DisconnectRes(reason="")


def create_simulated_clients(
    num_clients: int,
    model_fn: callable,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    device: torch.device,
    local_epochs: int = 2,
    learning_rate: float = 0.01,
    simulate_delay: bool = True,
    min_delay: float = 0.5,
    max_delay: float = 3.0,
    aggregation_strategy: str = "fedasync",
    strategy_params: Optional[Dict[str, Any]] = None,
) -> List[SimulatedAsyncClient]:
    """Create a list of simulated async clients.

    Args:
        num_clients: Number of clients to create
        model_fn: Function that returns a new model instance
        train_loaders: List of training data loaders (one per client)
        test_loaders: List of test data loaders (one per client)
        device: Torch device for training
        local_epochs: Number of local training epochs
        learning_rate: Learning rate for local training
        simulate_delay: Whether to simulate network delays
        min_delay: Minimum simulated delay in seconds
        max_delay: Maximum simulated delay in seconds
        aggregation_strategy: Async aggregation strategy name
        strategy_params: Strategy-specific parameters

    Returns:
        List of SimulatedAsyncClient instances
    """
    clients = []
    for i in range(num_clients):
        config = SimulatedClientConfig(
            client_id=str(i),
            model_fn=model_fn,
            train_loader=train_loaders[i % len(train_loaders)],
            test_loader=test_loaders[i % len(test_loaders)],
            device=device,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            simulate_delay=simulate_delay,
            min_delay=min_delay,
            max_delay=max_delay,
            aggregation_strategy=aggregation_strategy,
            strategy_params=strategy_params,
        )
        clients.append(SimulatedAsyncClient(config))
    return clients
