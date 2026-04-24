import time
import random

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import numpy as np

from algorithms.plora import (
    get_lora_parameters,
    inject_lora,
    set_lora_parameters,
)

class SyncSimulatedClient(fl.client.NumPyClient):
    """A clean, standard synchronous Flower client."""
    
    def __init__(self, cid: str, model_fn, train_loader, test_loader, device, cfg):
        self.cid = cid
        self.is_plora = str(cfg.server.strategy).lower() == "plora"

        model = model_fn()
        if self.is_plora:
            plora_cfg = cfg.get("plora", {})
            rank = plora_cfg.get("rank", 4)
            alpha = plora_cfg.get("alpha", 1.0)
            model = inject_lora(model, rank=rank, alpha=alpha)
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.cfg = cfg

        # Extracting network simulatino params
        self.simulate_delay = cfg.client.get("simulate_delay", False)
        self.min_delay = cfg.client.get("min_delay", 0.5)
        self.max_delay = cfg.client.get("max_delay", 3.0)

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Extract model weights as a list of NumPy arrays."""
        if self.is_plora:
            return get_lora_parameters(self.model)
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Load a list of NumPy arrays into the model weights."""
        if self.is_plora:
            set_lora_parameters(self.model, parameters)
            return
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Perform local training on the client's standard dataloader."""
        # Simulating Download
        if self.simulate_delay:
            download_delay = random.uniform(self.min_delay/2.0, self.max_delay/2.0)
            time.sleep(download_delay)
        self.set_parameters(parameters)
        self.model.train()
        
        if self.is_plora:
            trainable_params = [
                p
                for name, p in self.model.named_parameters()
                if "lora_" in name and p.requires_grad
            ]
        else:
            trainable_params = self.model.parameters()

        optimizer = optim.SGD(trainable_params, lr=self.cfg.client.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        num_examples = len(self.train_loader.dataset)
        
        for _ in range(self.cfg.client.local_epochs):
            for batch in self.train_loader:
                # Handle both flwr_datasets (dict) and standard PyTorch (tuple) batch structures
                if isinstance(batch, dict):
                    images = batch.get("img", batch.get("x")).to(self.device)
                    labels = batch.get("label", batch.get("y")).to(self.device)
                else:
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)

                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / max(1, len(self.train_loader) * self.cfg.client.local_epochs)

        # Simulate Upload Delay
        if self.simulate_delay:
            upload_delay = random.uniform(self.min_delay/2.0, self.max_delay /2.0)
            time.sleep(upload_delay)
        
        # Return updated weights, number of local samples, and metrics
        return self.get_parameters(config={}), num_examples, {"loss": avg_loss}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Perform local federated evaluation."""
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        num_examples = len(self.test_loader.dataset)
        total_loss, correct = 0.0, 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, dict):
                    images, labels = batch.get("img", batch.get("x")).to(self.device), batch.get("label", batch.get("y")).to(self.device)
                else:
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                    
                outputs = self.model(images)
                total_loss += criterion(outputs, labels).item() * labels.size(0)
                correct += outputs.max(1)[1].eq(labels).sum().item()

        avg_loss = float(total_loss / max(1, num_examples))
        accuracy = float(correct / max(1, num_examples))
        
        return avg_loss, num_examples, {"accuracy": accuracy}
