"""
STAMP: Spatio-Temporal Gradient Matching with Prototypical Coreset
arXiv 2025 - https://arxiv.org/abs/2506.12031
Reference code: (see paper and related repos)

Implements the STAMP algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy

class STAMPStrategy:
    def __init__(self, model: nn.Module, stamp_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            stamp_config: Configuration for STAMP (e.g., coreset size, gradient matching rules, etc.)
            num_clients: number of clients in federation
            kwargs: Additional STAMP-specific parameters.
        """
        self.model = model
        self.stamp_config = stamp_config
        self.num_clients = num_clients
        self.coreset_size = stamp_config.get('coreset_size', 100)
        # initialize prototypical coreset for each client
        self.client_coresets = [self.initialize_coreset() for _ in range(num_clients)]
        # store global coreset
        self.global_coreset = copy.deepcopy(self.client_coresets[0])

    def initialize_coreset(self):
        # initialize prototypical coreset for each client/task
        # for demonstration, use an empty list
        return []

    def get_coreset_params(self, client_id: int):
        # extract the current coreset parameters for a client
        return copy.deepcopy(self.client_coresets[client_id])

    def set_coreset_params(self, coreset_params: list, client_id: int):
        # set the coreset parameters for a client
        self.client_coresets[client_id] = copy.deepcopy(coreset_params)

    def aggregate(self, client_coresets: List[list]):
        # aggregate client coreset parameters (FedAvg for coreset samples)
        # for demonstration, just concatenate and sample up to coreset_size
        all_samples = []
        for cs in client_coresets:
            all_samples.extend(cs)
        if len(all_samples) > self.coreset_size:
            indices = torch.randperm(len(all_samples))[:self.coreset_size]
            agg_coreset = [all_samples[i] for i in indices]
        else:
            agg_coreset = all_samples
        self.global_coreset = agg_coreset
        return agg_coreset

    def train_round(self, data, task_id: int, client_id: int):
        # perform a training round for a given task (coreset-based local training)
        # update coreset with new samples
        for x, y in data:
            if len(self.client_coresets[client_id]) < self.coreset_size:
                self.client_coresets[client_id].append((x.clone(), y.clone()))
            else:
                idx = torch.randint(0, self.coreset_size, (1,)).item()
                self.client_coresets[client_id][idx] = (x.clone(), y.clone())
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        for x, y in data:
            optimizer.zero_grad()
            out = self.model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            optimizer.step()
        return self.get_coreset_params(client_id)

    def evaluate(self, data):
        # evaluate the model on the given data
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data:
                out = self.model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0 