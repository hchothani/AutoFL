"""Simulated async client for async federated learning."""

from __future__ import annotations

import time
import json
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

@dataclass
class SimulatedClientConfig:
    client_id: str
    model_fn: callable 
    train_loaders: List[DataLoader]
    test_loaders: List[DataLoader]
    device: torch.device
    local_epochs: int = 2
    learning_rate: float = 0.01
    simulate_delay: bool = True
    min_delay: float = 0.5
    max_delay: float = 3.0

class SimulatedAsyncClient(ClientProxy):
    def __init__(self, config: SimulatedClientConfig):
        super().__init__(config.client_id)
        self.config = config
        self.model = config.model_fn().to(config.device)
        self.train_loaders = config.train_loaders
        self.test_loaders = config.test_loaders
        self.device = config.device
        self.local_epochs = config.local_epochs
        self.learning_rate = config.learning_rate

    def get_parameters(self, ins: GetParametersIns, timeout: Optional[float] = None) -> GetParametersRes:
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(params),
        )

    def get_properties(self, ins, timeout: Optional[float] = None):
        from flwr.common import GetPropertiesRes
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={
                "client_id": self.cid,
                "num_examples": len(self.train_loaders[0].dataset), # Fallback estimate
            },
        )

    def fit(self, ins: FitIns, timeout: Optional[float] = None) -> FitRes:
        start_time = time.time()
        current_phase = ins.config.get("current_phase", 0)
        active_train_loader = self.train_loaders[current_phase]
        num_examples = len(active_train_loader.dataset)

        # print(f" -> [Vehicle {self.cid}] Async Trigger: Phase {current_phase} training on {num_examples} images.")

        if self.config.simulate_delay:
            download_delay = random.uniform(self.config.min_delay / 2, self.config.max_delay / 2)
            time.sleep(download_delay)

        params = parameters_to_ndarrays(ins.parameters)
        state_dict = self.model.state_dict()
        for key, param in zip(state_dict.keys(), params):
            state_dict[key] = torch.tensor(param)
        self.model.load_state_dict(state_dict)

        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0
        for epoch in range(self.local_epochs):
            for batch in active_train_loader:
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

        # Prototype Extraction (Post-Training)
        self.model.eval()

        feature_sum = None
        feature_count = 0

        def feature_hook(module, input, output):
            nonlocal feature_sum, feature_count
            # input[0] shape: (batch_size, feature_dim)
            batch_features = input[0].detach().cpu().numpy()

            if feature_sum is None:
                feature_sum = np.sum(batch_features, axis=0)
            else:
                feature_sum += np.sum(batch_features, axis=0)
            feature_count += batch_features.shape[0]

        # Attach Hook
        last_linear_layer = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear): last_linear_layer = module
        if last_linear_layer is None: last_linear_layer = list(self.model.children())[-1]
            
        hook_handle = last_linear_layer.register_forward_hook(feature_hook)

        # Single Rapid Forward Pass without gradients
        try:
            with torch.no_grad():
                for batch in active_train_loader:
                    if isinstance(batch, dict):
                        images = batch.get("img", batch.get("x")).to(self.device)
                    elif isinstance(batch, (tuple, list)):
                        images = batch[0].to(self.device)
                    else: continue
                    _ = self.model(images)
        finally:
            hook_handle.remove() # Always clean up

        # Calculate perfect centroid
        if feature_count > 0:
            client_prototype = (feature_sum / feature_count).tolist()
        else:
            client_prototype = None

        if self.config.simulate_delay:
            upload_delay = random.uniform(self.config.min_delay / 2, self.config.max_delay / 2)
            time.sleep(upload_delay)

        new_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        elapsed = time.time() - start_time

        metrics={
                "loss": avg_loss,
                "training_time": elapsed,
                "start_timestamp": ins.config.get("start_timestamp", start_time),
                "client_id": self.cid,
        }

        if client_prototype is not None:
            metrics["prototype"] = json.dumps(client_prototype)

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(new_params),
            num_examples=num_examples,
            metrics=metrics
        )

    def evaluate(self, ins: EvaluateIns, timeout: Optional[float] = None) -> EvaluateRes:
        current_phase = ins.config.get("current_phase", 0)
        active_test_loader = self.test_loaders[current_phase]
        
        params = parameters_to_ndarrays(ins.parameters)
        state_dict = self.model.state_dict()
        for key, param in zip(state_dict.keys(), params):
            state_dict[key] = torch.tensor(param)
        self.model.load_state_dict(state_dict)

        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in active_test_loader:
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
        from flwr.common import DisconnectRes
        return DisconnectRes(reason="")

def create_simulated_clients(
    num_clients: int,
    model_fn: callable,
    train_loaders: List[List[DataLoader]],
    test_loaders: List[List[DataLoader]],
    device: torch.device,
    local_epochs: int = 2,
    learning_rate: float = 0.01,
    simulate_delay: bool = True,
    min_delay: float = 0.5,
    max_delay: float = 3.0,
) -> List[SimulatedAsyncClient]:
    clients = []
    for i in range(num_clients):
        config = SimulatedClientConfig(
            client_id=str(i),
            model_fn=model_fn,
            train_loaders=train_loaders[i],
            test_loaders=test_loaders[i],
            device=device,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            simulate_delay=simulate_delay,
            min_delay=min_delay,
            max_delay=max_delay,
        )
        clients.append(SimulatedAsyncClient(config))
    return clients
