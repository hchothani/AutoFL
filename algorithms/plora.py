"""
PLoRA: Parameter-Efficient LoRA-based Federated Continual Learning
ICLR 2024 submission - https://arxiv.org/abs/2401.02094
Reference code: (see paper and related repos)

Implements the PLoRA algorithm for Federated Continual Learning (LoRA-based adaptation).
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy

class PLoRAStrategy:
    def __init__(self, model: nn.Module, plora_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            plora_config: Configuration for LoRA adaptation (e.g., rank, layers, etc.)
            num_clients: number of clients in federation
            kwargs: Additional PLoRA-specific parameters.
        """
        self.model = model
        self.plora_config = plora_config
        self.num_clients = num_clients
        # initialize LoRA modules for each client
        self.client_loras = [self.initialize_lora() for _ in range(num_clients)]
        # store global LoRA parameters
        self.global_lora = copy.deepcopy(self.client_loras[0])

    def initialize_lora(self):
        # initialize LoRA modules for parameter-efficient adaptation
        rank = self.plora_config.get('rank', 4)
        in_dim = self.plora_config.get('in_dim', 128)
        out_dim = self.plora_config.get('out_dim', 128)
        # create low-rank matrices a and b
        a = torch.nn.Parameter(torch.randn(out_dim, rank))
        b = torch.nn.Parameter(torch.randn(rank, in_dim))
        return {'a': a, 'b': b}

    def get_lora_params(self):
        # extract the current LoRA parameters from the model
        return {'a': self.model.lora_a.data.clone(), 'b': self.model.lora_b.data.clone()}

    def set_lora_params(self, lora_params: Dict[str, Any]):
        # set the LoRA parameters in the model
        self.model.lora_a.data.copy_(lora_params['a'])
        self.model.lora_b.data.copy_(lora_params['b'])

    def aggregate(self, client_loras: List[Dict[str, Any]]):
        # aggregate client LoRA parameters (FedAvg)
        a_stack = torch.stack([cl['a'] for cl in client_loras], dim=0)
        b_stack = torch.stack([cl['b'] for cl in client_loras], dim=0)
        agg_a = torch.mean(a_stack, dim=0)
        agg_b = torch.mean(b_stack, dim=0)
        self.global_lora = {'a': agg_a, 'b': agg_b}
        return self.global_lora

    def train_round(self, data, task_id: int, client_id: int):
        # perform a training round for a given task (LoRA-based local training)
        self.set_lora_params(self.client_loras[client_id])
        self.model.train()
        optimizer = torch.optim.Adam([self.model.lora_a, self.model.lora_b], lr=0.001)
        for x, y in data:
            optimizer.zero_grad()
            out = self.model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            optimizer.step()
        self.client_loras[client_id] = self.get_lora_params()
        return self.get_lora_params()

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