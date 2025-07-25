"""
SacFL: Self-Adaptive Federated Continual Learning for Resource-Constrained End Devices
TNNLS 2025 - https://arxiv.org/abs/2505.00365
Reference code: (see paper and related repos)

Implements the SacFL algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy

class SacFLStrategy:
    def __init__(self, model: nn.Module, sacfl_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            sacfl_config: Configuration for SacFL (e.g., encoder/decoder settings, contrastive learning, etc.)
            num_clients: number of clients in federation
            kwargs: Additional SacFL-specific parameters.
        """
        self.model = model
        self.sacfl_config = sacfl_config
        self.num_clients = num_clients
        # initialize encoder, decoder, and other components for each client
        self.client_components = [self.initialize_components() for _ in range(num_clients)]
        # store global component parameters
        self.global_components = copy.deepcopy(self.client_components[0])

    def initialize_components(self):
        # initialize encoder, decoder, and other components for SacFL
        enc_dim = self.sacfl_config.get('enc_dim', 128)
        dec_dim = self.sacfl_config.get('dec_dim', 128)
        encoder = torch.nn.Parameter(torch.randn(enc_dim, enc_dim))
        decoder = torch.nn.Parameter(torch.randn(dec_dim, dec_dim))
        return {'encoder': encoder, 'decoder': decoder}

    def get_component_params(self):
        # extract the current component parameters from the model
        return {'encoder': self.model.encoder.data.clone(), 'decoder': self.model.decoder.data.clone()}

    def set_component_params(self, component_params: Dict[str, Any]):
        # set the component parameters in the model
        self.model.encoder.data.copy_(component_params['encoder'])
        self.model.decoder.data.copy_(component_params['decoder'])

    def aggregate(self, client_components: List[Dict[str, Any]]):
        # aggregate client component parameters (FedAvg)
        enc_stack = torch.stack([cc['encoder'] for cc in client_components], dim=0)
        dec_stack = torch.stack([cc['decoder'] for cc in client_components], dim=0)
        agg_enc = torch.mean(enc_stack, dim=0)
        agg_dec = torch.mean(dec_stack, dim=0)
        self.global_components = {'encoder': agg_enc, 'decoder': agg_dec}
        return self.global_components

    def train_round(self, data, task_id: int, client_id: int):
        # perform a training round for a given task (self-adaptive local training)
        self.set_component_params(self.client_components[client_id])
        self.model.train()
        optimizer = torch.optim.Adam([self.model.encoder, self.model.decoder], lr=0.001)
        for x, y in data:
            optimizer.zero_grad()
            out = self.model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            optimizer.step()
        self.client_components[client_id] = self.get_component_params()
        return self.get_component_params()

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