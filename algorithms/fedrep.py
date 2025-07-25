"""
FedRep: Federated Learning with Personalization Layers
ICLR 2022 - https://arxiv.org/abs/2102.07078
Reference code: https://github.com/luopanyaxin/Federated-Continual-Learning

Implements the FedRep algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional, Callable
import copy

class FedRepStrategy:
    def __init__(self, model: nn.Module, rep_layer_names: List[str], device: str = 'cpu', **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            rep_layer_names: List of layer names to be treated as the shared representation.
            device: Device to run the model on ('cpu' or 'cuda').
            kwargs: Additional FedRep-specific parameters.
        """
        self.model = model.to(device)
        self.rep_layer_names = rep_layer_names
        self.device = device
        self.global_rep = self._get_representation()
        self.client_classifiers = {}
        # Optional hooks for continual learning
        self.on_task_start: Optional[Callable] = kwargs.get('on_task_start', None)
        self.on_task_end: Optional[Callable] = kwargs.get('on_task_end', None)

    def _get_named_params(self, names: List[str]):
        # get parameters by name
        return {k: v.clone().detach() for k, v in self.model.state_dict().items() if k in names}

    def _get_other_params(self, names: List[str]):
        # get parameters not in names
        return {k: v.clone().detach() for k, v in self.model.state_dict().items() if k not in names}

    def _set_named_params(self, params: Dict[str, Any]):
        # set parameters by name
        state = self.model.state_dict()
        for k, v in params.items():
            state[k] = v.clone().detach()
        self.model.load_state_dict(state)

    def get_representation(self):
        # extract the shared representation parameters from the model
        return self._get_named_params(self.rep_layer_names)

    def get_classifier(self):
        # extract the local classifier parameters from the model
        return self._get_other_params(self.rep_layer_names)

    def set_representation(self, rep_params: Dict[str, Any]):
        # set the shared representation parameters in the model
        self._set_named_params(rep_params)

    def set_classifier(self, clf_params: Dict[str, Any]):
        # set the local classifier parameters in the model
        self._set_named_params(clf_params)

    def freeze_layers(self, freeze_names: List[str]):
        # freeze parameters by name
        for name, param in self.model.named_parameters():
            param.requires_grad = name in freeze_names

    def unfreeze_layers(self, unfreeze_names: List[str]):
        # unfreeze only the given names, freeze the rest
        for name, param in self.model.named_parameters():
            param.requires_grad = name in unfreeze_names

    def aggregate(self, client_reps: List[Dict[str, Any]]):
        # aggregate client representations (FedAvg)
        agg_rep = {}
        for k in self.rep_layer_names:
            vals = [client[k] for client in client_reps]
            stacked = torch.stack(vals, dim=0)
            agg_rep[k] = torch.mean(stacked, dim=0)
        self.global_rep = agg_rep
        return agg_rep

    def train_round(self, data, task_id: int, client_id: int,
                    local_head_epochs: int = 1, local_rep_epochs: int = 1,
                    lr: float = 0.01, optimizer_cls=torch.optim.SGD):
        """
        Perform a local training round for a given task (FedRep two-phase training):
        1. Freeze rep, train head for local_head_epochs
        2. Freeze head, train rep for local_rep_epochs
        """
        self.model.to(self.device)
        self.set_representation(self.global_rep)
        if client_id in self.client_classifiers:
            self.set_classifier(self.client_classifiers[client_id])
        # Optional continual learning hook
        if self.on_task_start:
            self.on_task_start(task_id)
        # Phase 1: Train head (freeze rep)
        self.freeze_layers([n for n in self.model.state_dict().keys() if n not in self.rep_layer_names])
        self.unfreeze_layers([n for n in self.model.state_dict().keys() if n not in self.rep_layer_names])
        optimizer = optimizer_cls(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.model.train()
        for _ in range(local_head_epochs):
            for x, y in data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                loss.backward()
                optimizer.step()
        # Phase 2: Train rep (freeze head)
        self.freeze_layers(self.rep_layer_names)
        self.unfreeze_layers(self.rep_layer_names)
        optimizer = optimizer_cls(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        for _ in range(local_rep_epochs):
            for x, y in data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                loss.backward()
                optimizer.step()
        # Update local classifier
        self.client_classifiers[client_id] = self.get_classifier()
        # Optional continual learning hook
        if self.on_task_end:
            self.on_task_end(task_id)
        return self.get_representation(), self.get_classifier()

    def evaluate(self, data):
        # evaluate the model on the given data
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0 