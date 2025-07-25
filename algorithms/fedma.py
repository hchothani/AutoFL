"""
FedMA: Federated Matched Averaging
NeurIPS 2020 - https://arxiv.org/abs/2002.06440
Reference code: https://github.com/IBM/FedMA

Implements the FedMA algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy

class FedMAStrategy:
    def __init__(self, model: nn.Module, ma_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            ma_config: Configuration for matched averaging (e.g., matching rules, layers, etc.)
            num_clients: number of clients in federation
            kwargs: Additional FedMA-specific parameters.
        """
        self.model = model
        self.ma_config = ma_config
        self.num_clients = num_clients
        # store client models
        self.client_models = [copy.deepcopy(model) for _ in range(num_clients)]
        # store global model
        self.global_model = copy.deepcopy(model)

    def match_layers(self, client_models: List[nn.Module]):
        # perform layer-wise matching across client models
        # for demonstration, use simple averaging for each layer
        matched_state = {}
        state_dicts = [cm.state_dict() for cm in client_models]
        for k in state_dicts[0].keys():
            vals = [sd[k] for sd in state_dicts]
            stacked = torch.stack(vals, dim=0)
            matched_state[k] = torch.mean(stacked, dim=0)
        return matched_state

    def aggregate(self, matched_models: List[nn.Module]):
        # aggregate matched client models (FedMA logic)
        matched_state = self.match_layers(matched_models)
        self.global_model.load_state_dict(matched_state)
        return self.global_model

    def train_round(self, data, task_id: int, client_id: int):
        # perform a training round for a given task (FedMA-based local training)
        model = self.client_models[client_id]
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for x, y in data:
            optimizer.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            optimizer.step()
        self.client_models[client_id] = model
        return model

    def evaluate(self, data):
        # evaluate the global model on the given data
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data:
                out = self.global_model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0 