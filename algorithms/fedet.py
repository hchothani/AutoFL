"""
FedET: Federated Enhanced Transformer for Class-Incremental Learning
IJCAI 2023 - https://arxiv.org/abs/2305.18213
Reference code: https://github.com/luopanyaxin/Federated-Continual-Learning (see also FedL2P, DualPrompt)

Implements the FedET algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy

class FedETStrategy:
    def __init__(self, model: nn.Module, et_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            et_config: Configuration for enhanced transformer (e.g., prompt layers, attention heads, etc.)
            num_clients: number of clients in federation
            kwargs: Additional FedET-specific parameters.
        """
        self.model = model
        self.et_config = et_config
        self.num_clients = num_clients
        # initialize transformer-specific parameters for each client
        self.client_transformers = [self.initialize_transformer() for _ in range(num_clients)]
        # store global transformer parameters
        self.global_transformer = copy.deepcopy(self.client_transformers[0])

    def initialize_transformer(self):
        # initialize transformer-specific parameters (e.g., prompts, heads)
        # for demonstration, assume a prompt tensor
        prompt_len = self.et_config.get('prompt_length', 10)
        prompt_dim = self.et_config.get('prompt_dim', 128)
        prompt = torch.nn.Parameter(torch.randn(prompt_len, prompt_dim))
        return {'prompt': prompt}

    def get_transformer_params(self):
        # extract the current transformer parameters from the model
        return {'prompt': self.model.prompt.data.clone()}

    def set_transformer_params(self, transformer_params: Dict[str, Any]):
        # set the transformer parameters in the model
        self.model.prompt.data.copy_(transformer_params['prompt'])

    def aggregate(self, client_transformers: List[Dict[str, Any]]):
        # aggregate client transformer parameters (FedAvg)
        stacked = torch.stack([ct['prompt'] for ct in client_transformers], dim=0)
        agg_prompt = torch.mean(stacked, dim=0)
        self.global_transformer = {'prompt': agg_prompt}
        return self.global_transformer

    def train_round(self, data, task_id: int, client_id: int):
        # perform a training round for a given task (transformer-based local training)
        self.set_transformer_params(self.client_transformers[client_id])
        self.model.train()
        optimizer = torch.optim.Adam([self.model.prompt], lr=0.001)
        for x, y in data:
            optimizer.zero_grad()
            out = self.model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            optimizer.step()
        self.client_transformers[client_id] = self.get_transformer_params()
        return self.get_transformer_params()

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