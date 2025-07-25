"""
FedRCIL: Federated Knowledge Distillation for Representation-based Contrastive Incremental Learning
ICCV Workshop 2023 - https://arxiv.org/abs/2308.09941
Reference code: (see paper and related repos)

Implements the FedRCIL algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy

class FedRCILStrategy:
    def __init__(self, model: nn.Module, rc_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            rc_config: Configuration for FedRCIL (e.g., distillation settings, contrastive loss, etc.)
            num_clients: number of clients in federation
            kwargs: Additional FedRCIL-specific parameters.
        """
        self.model = model
        self.rc_config = rc_config
        self.num_clients = num_clients
        # initialize distillation and contrastive learning components for each client
        self.client_distills = [self.initialize_distillation() for _ in range(num_clients)]
        # store global distillation parameters
        self.global_distill = copy.deepcopy(self.client_distills[0])

    def initialize_distillation(self):
        # initialize distillation and contrastive learning components
        # for demonstration, assume a temperature parameter and a projection head
        temperature = self.rc_config.get('temperature', 0.5)
        proj_dim = self.rc_config.get('proj_dim', 128)
        projection = torch.nn.Parameter(torch.randn(proj_dim, proj_dim))
        return {'temperature': temperature, 'projection': projection}

    def get_distillation_params(self):
        # extract the current distillation parameters from the model
        return {'temperature': self.rc_config.get('temperature', 0.5), 'projection': self.model.projection.data.clone()}

    def set_distillation_params(self, distill_params: Dict[str, Any]):
        # set the distillation parameters in the model
        self.model.projection.data.copy_(distill_params['projection'])
        self.rc_config['temperature'] = distill_params['temperature']

    def aggregate(self, client_distills: List[Dict[str, Any]]):
        # aggregate client distillation parameters (FedAvg for projection, mean for temperature)
        temps = [cd['temperature'] for cd in client_distills]
        projs = [cd['projection'] for cd in client_distills]
        agg_temp = sum(temps) / len(temps)
        stacked = torch.stack(projs, dim=0)
        agg_proj = torch.mean(stacked, dim=0)
        self.global_distill = {'temperature': agg_temp, 'projection': agg_proj}
        return self.global_distill

    def train_round(self, data, task_id: int, client_id: int):
        # perform a training round for a given task (distillation-based local training)
        self.set_distillation_params(self.client_distills[client_id])
        self.model.train()
        optimizer = torch.optim.Adam([self.model.projection], lr=0.001)
        temperature = self.rc_config.get('temperature', 0.5)
        for x, y in data:
            optimizer.zero_grad()
            out = self.model(x)
            # contrastive loss (simplified)
            proj_out = torch.matmul(out, self.model.projection)
            logits = proj_out / temperature
            labels = y
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
        self.client_distills[client_id] = self.get_distillation_params()
        return self.get_distillation_params()

    def evaluate(self, data):
        # evaluate the model on the given data
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data:
                out = self.model(x)
                proj_out = torch.matmul(out, self.model.projection)
                pred = proj_out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0 