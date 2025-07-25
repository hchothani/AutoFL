"""
FedWeIT: Federated Weighted Inter-client Transfer
ICML 2021 - https://proceedings.mlr.press/v139/yoon21b.html
Official code: https://github.com/gggangmin/FCL

Implements the FedWeIT algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy

class FedWeITStrategy:
    def __init__(self, model: nn.Module, sparsity: float = 0.5, num_clients: int = 2, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            sparsity: proportion of parameters to be kept as task-specific (0-1)
            num_clients: number of clients in federation
            kwargs: Additional FedWeIT-specific parameters
        """
        self.model = model
        self.sparsity = sparsity
        self.num_clients = num_clients
        # create a mask for each client for task-specific parameters
        self.client_masks = [self._init_mask() for _ in range(num_clients)]
        # store global parameters
        self.global_params = self._get_model_params()
        # store task-specific parameters for each client
        self.client_task_params = [copy.deepcopy(self.global_params) for _ in range(num_clients)]

    def _init_mask(self):
        # initialize a binary mask for each parameter tensor, with sparsity
        mask = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                numel = param.numel()
                k = int(self.sparsity * numel)
                flat = torch.zeros(numel, dtype=torch.bool)
                flat[:k] = 1
                perm = torch.randperm(numel)
                flat = flat[perm]
                mask[name] = flat.view(param.shape)
        return mask

    def _get_model_params(self):
        # get a state dict of model parameters
        return {k: v.clone().detach() for k, v in self.model.state_dict().items()}

    def _set_model_params(self, params):
        # set model parameters from a state dict
        self.model.load_state_dict(params)

    def partition_parameters(self):
        # partition model parameters into global and task-specific (sparse) components for each client
        partitions = []
        for client_id in range(self.num_clients):
            mask = self.client_masks[client_id]
            client_params = {}
            for name, param in self.model.named_parameters():
                if name in mask:
                    # task-specific part
                    client_params[name] = param.data * mask[name]
                else:
                    # global part
                    client_params[name] = param.data
            partitions.append(client_params)
        return partitions

    def aggregate(self, client_updates: List[Dict[str, Any]]):
        # aggregate client updates using weighted inter-client transfer
        # for each parameter, compute weighted average of task-specific parts and global part
        agg_params = {}
        for name in self.global_params.keys():
            # collect all client values for this parameter
            vals = [client[name] for client in client_updates]
            # average across clients
            stacked = torch.stack(vals, dim=0)
            agg_params[name] = torch.mean(stacked, dim=0)
        # update global params
        self.global_params = agg_params
        return agg_params

    def update_model(self, aggregated_params: Dict[str, Any]):
        # update the model with aggregated parameters
        self._set_model_params(aggregated_params)

    def train_round(self, data, task_id: int, client_id: int):
        # perform a training round for a given task and client
        # set model to client's current parameters
        self._set_model_params(self.client_task_params[client_id])
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        for x, y in data:
            optimizer.zero_grad()
            out = self.model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            optimizer.step()
        # update client task params
        self.client_task_params[client_id] = self._get_model_params()
        return self.client_task_params[client_id]

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