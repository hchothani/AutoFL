import time
import random

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import numpy as np

class SyncSimulatedClient(fl.client.NumPyClient):
    """A clean, standard synchronous Flower client."""
    
    def __init__(self, cid: str, model_fn, train_loaders: list, test_loaders: list, device, cfg):
        self.cid = cid
        self.model = model_fn().to(device)
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.device = device
        self.cfg = cfg

        # Extracting network simulation params
        self.simulate_delay = cfg.client.get("simulate_delay", False)
        self.min_delay = cfg.client.get("min_delay", 0.5)
        self.max_delay = cfg.client.get("max_delay", 3.0)

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        # Dynamically select phase loader
        current_phase = config.get("current_phase", 0)
        active_train_loader = self.train_loaders[current_phase]
        
        if self.simulate_delay:
            download_delay = random.uniform(self.min_delay/2.0, self.max_delay/2.0)
            time.sleep(download_delay)
            
        self.set_parameters(parameters)
        self.model.train()
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.cfg.client.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        num_examples = len(active_train_loader.dataset)
        
        for _ in range(self.cfg.client.local_epochs):
            for batch in active_train_loader:
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

        avg_loss = total_loss / max(1, len(active_train_loader) * self.cfg.client.local_epochs)

        self.model.eval() # Lock weights and batch norm
        
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

        # Single rapid forward pass without gradients
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

        if self.simulate_delay:
            upload_delay = random.uniform(self.min_delay/2.0, self.max_delay /2.0)
            time.sleep(upload_delay)

        metrics = {
            "loss": avg_loss,
            "prototype": client_prototype
        }
        
        return self.get_parameters(config={}), num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        current_phase = config.get("current_phase", 0)
        active_test_loader = self.test_loaders[current_phase]
        
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        num_examples = len(active_test_loader.dataset)
        total_loss, correct = 0.0, 0
        
        with torch.no_grad():
            for batch in active_test_loader:
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
