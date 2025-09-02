"""
fedweit: federated weighted inter-client transfer
icml 2021 - https://proceedings.mlr.press/v139/yoon21b.html
official code: https://github.com/gggangmin/FCL

implements the fedweit algorithm for federated continual learning.
this class is designed to be compatible with avalanche/flwr integration.
"""

import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np

# Flower imports for custom server strategy
from flwr.server.strategy import Strategy
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy

class FedWeITStrategy:
    def __init__(self, model: nn.Module, sparsity: float = 0.5, num_clients: int = 2, l1_lambda: float = 0.1, l2_lambda: float = 100.0, device: str = 'cpu', **kwargs):
        """
        args:
            model: pytorch model to be used for federated continual learning.
            sparsity: target sparsity for task-adaptive parameters (0-1)
            num_clients: number of clients in federation
            l1_lambda: l1 regularization for sparsity
            l2_lambda: retroactive update regularization
            device: device to run model on
            kwargs: additional fedweit-specific parameters
        """
        self.model = model.to(device)
        self.sparsity = sparsity
        self.num_clients = num_clients
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.device = device
        # initialize base parameters (shared across all tasks/clients)
        self.base_params = self._get_model_params()
        # initialize per-client, per-task mask and task-adaptive params
        self.client_task_masks = [{} for _ in range(num_clients)]  # list of dicts: client -> {task_id: mask}
        self.client_task_adaptives = [{} for _ in range(num_clients)]  # list of dicts: client -> {task_id: task-adaptive param}
        # initialize attention weights for inter-client transfer
        self.client_task_attn = [{} for _ in range(num_clients)]  # list of dicts: client -> {task_id: attn vector}
        # knowledge base: stores all task-adaptive params for all clients/tasks
        self.knowledge_base = []  # list of (client_id, task_id, task-adaptive param)
        
        # state tracking for avalanche interface
        self.current_client_id = 0  # will be set dynamically
        self.current_task_id = 0   # will be set dynamically

    def _get_model_params(self):
        # get a state dict of model parameters
        return {k: v.clone().detach().to(self.device) for k, v in self.model.state_dict().items()}

    def _set_model_params(self, params):
        # set model parameters from a state dict
        self.model.load_state_dict(params)

    def _init_mask(self, param, sparsity):
        # initialize a learnable mask for a parameter tensor
        numel = param.numel()
        k = int(sparsity * numel)
        mask = torch.zeros(numel, device=self.device)
        mask[:k] = 1.0
        perm = torch.randperm(numel)
        mask = mask[perm].view(param.shape)
        mask = nn.Parameter(mask, requires_grad=True)
        return mask

    def _init_task_adaptive(self, param):
        # initialize a sparse task-adaptive parameter (same shape, zeros)
        return nn.Parameter(torch.zeros_like(param, device=self.device), requires_grad=True)

    def _init_attn(self, num_kb):
        # initialize attention weights over knowledge base (softmaxed)
        attn = torch.ones(num_kb, device=self.device) / max(1, num_kb)
        attn = nn.Parameter(attn, requires_grad=True)
        return attn

    def decompose_params(self, base_params, mask, task_adaptive, kb_attn=None, kb_params=None):
        # reconstruct model params from base, mask, task-adaptive, and optionally weighted kb params
        params = {}
        for name in base_params:
            p = base_params[name] * mask[name] + task_adaptive[name]
            if kb_attn is not None and kb_params is not None:
                # weighted sum of kb task-adaptive params
                kb_sum = sum(w * kb[name] for w, kb in zip(kb_attn, kb_params))
                p = p + kb_sum
            params[name] = p
        return params

    def train_task(self, data, task_id: int, client_id: int, epochs: int = 1, lr: float = 1e-3):
        # train on a single task for a client
        # initialize mask and task-adaptive if not present
        if task_id not in self.client_task_masks[client_id]:
            self.client_task_masks[client_id][task_id] = {}
            self.client_task_adaptives[client_id][task_id] = {}
            for name, param in self.model.named_parameters():
                self.client_task_masks[client_id][task_id][name] = self._init_mask(param, self.sparsity)
                self.client_task_adaptives[client_id][task_id][name] = self._init_task_adaptive(param)
        mask = self.client_task_masks[client_id][task_id]
        task_adaptive = self.client_task_adaptives[client_id][task_id]
        # get relevant kb params and attn
        kb_params = [kb[2] for kb in self.knowledge_base if kb[0] != client_id]
        if len(kb_params) > 0:
            if task_id not in self.client_task_attn[client_id]:
                self.client_task_attn[client_id][task_id] = self._init_attn(len(kb_params))
            attn = F.softmax(self.client_task_attn[client_id][task_id], dim=0)
        else:
            attn = None
        # set model params
        params = self.decompose_params(self.base_params, mask, task_adaptive, attn, kb_params if attn is not None else None)
        self._set_model_params(params)
        # collect all learnable params
        learnable = list(mask.values()) + list(task_adaptive.values())
        if attn is not None:
            learnable.append(self.client_task_attn[client_id][task_id])
        optimizer = Adam(learnable, lr=lr)
        for epoch in range(epochs):
            self.model.train()
            for x, y in data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                # l1 regularization for mask and task-adaptive
                l1 = sum(torch.abs(m).sum() for m in mask.values()) + sum(torch.abs(a).sum() for a in task_adaptive.values())
                # retroactive l2 update (not fully implemented, placeholder)
                l2 = 0.0
                loss = loss + self.l1_lambda * l1 + self.l2_lambda * l2
                loss.backward()
                optimizer.step()
        # update knowledge base
        kb_entry = (client_id, task_id, {k: v.detach().clone() for k, v in task_adaptive.items()})
        self.knowledge_base.append(kb_entry)
        # update base params (aggregate at server in real use)
        # here, just keep as is
        return self.get_sparse_update(client_id, task_id)

    def get_sparse_update(self, client_id, task_id):
        # return only the masked base params and task-adaptive params for communication
        mask = self.client_task_masks[client_id][task_id]
        task_adaptive = self.client_task_adaptives[client_id][task_id]
        masked_base = {k: self.base_params[k] * mask[k].detach() for k in self.base_params}
        task_adaptive = {k: v.detach() for k, v in task_adaptive.items()}
        return {'masked_base': masked_base, 'task_adaptive': task_adaptive}

    def aggregate(self, client_updates: List[Dict[str, Any]]):
        # aggregate masked base params from all clients (server-side)
        agg_params = {}
        for name in self.base_params:
            vals = [cu['masked_base'][name] for cu in client_updates]
            stacked = torch.stack(vals, dim=0)
            agg_params[name] = torch.mean(stacked, dim=0)
        self.base_params = agg_params
        return agg_params

    def evaluate(self, data, task_id: int, client_id: int):
        # evaluate the model for a given client/task
        mask = self.client_task_masks[client_id][task_id]
        task_adaptive = self.client_task_adaptives[client_id][task_id]
        kb_params = [kb[2] for kb in self.knowledge_base if kb[0] != client_id]
        if len(kb_params) > 0 and task_id in self.client_task_attn[client_id]:
            attn = F.softmax(self.client_task_attn[client_id][task_id], dim=0)
        else:
            attn = None
        params = self.decompose_params(self.base_params, mask, task_adaptive, attn, kb_params if attn is not None else None)
        self._set_model_params(params)
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

    # avalanche interface methods
    def train(self, experience):
        """
        train method that matches avalanche interface.
        extracts task_id and client_id from the experience and calls train_task.
        """
        # extract task id from experience
        task_id = experience.current_experience
        
        # for federated learning, we need to determine client_id
        # since we don't have direct access to client_id from experience,
        # we'll use the current_client_id that should be set externally
        client_id = self.current_client_id
        
        # extract data from experience
        # experience.dataset is an avalanche dataset, we need to create a dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(experience.dataset, batch_size=32, shuffle=True)
        
        print(f"[FedWeIT] Training on experience {task_id} for client {client_id}")
        
        # call the original train_task method
        return self.train_task(dataloader, task_id, client_id, epochs=1, lr=1e-3)
    
    def eval(self, test_stream):
        """
        eval method that matches avalanche interface.
        evaluates on all experiences in the test stream.
        """
        results = {}
        
        # get current client_id
        client_id = self.current_client_id
        
        print(f"[FedWeIT] Evaluating client {client_id} on test stream")
        
        # iterate through all experiences in test stream
        for experience in test_stream:
            task_id = experience.current_experience
            
            # skip if we haven't trained on this task yet
            if task_id not in self.client_task_masks[client_id]:
                print(f"[FedWeIT] Skipping evaluation on task {task_id} - not trained yet")
                continue
            
            # create dataloader from experience dataset
            from torch.utils.data import DataLoader
            dataloader = DataLoader(experience.dataset, batch_size=32, shuffle=False)
            
            # evaluate on this experience
            accuracy = self.evaluate(dataloader, task_id, client_id)
            
            # store results in the format expected by avalanche
            results[f"Top1_Acc_Exp/eval_phase/test_stream/Task{task_id:03d}"] = accuracy
            
        print(f"[FedWeIT] Evaluation results: {results}")
        return results

class FedWeITServerStrategy:
    """
    Custom server aggregation logic for FedWeIT algorithm.
    Implements server-side aggregation of masked base parameters and task-adaptive parameters.
    
    Note: This is a simplified server strategy that implements the core FedWeIT aggregation logic.
    In a full implementation, this would inherit from flwr.server.strategy.Strategy.
    """
    
    def __init__(self, **kwargs):
        # FedWeIT-specific state
        self.base_params = None
        self.round = 0
        print("[FedWeIT Server] Initialized FedWeIT server aggregation strategy")
    
    def aggregate_client_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Core FedWeIT aggregation logic: average masked base parameters.
        
        Args:
            client_updates: List of dicts containing 'masked_base' and 'task_adaptive' from clients
            
        Returns:
            Aggregated parameters dict
        """
        if not client_updates:
            return {}
        
        print(f"[FedWeIT Server] Aggregating {len(client_updates)} client updates")
        
        # FedWeIT aggregation: average masked base parameters
        agg_params = {}
        first_update = client_updates[0]
        
        if 'masked_base' in first_update:
            agg_params['masked_base'] = {}
            for param_name in first_update['masked_base']:
                # Collect all client masked_base params for this layer
                param_values = []
                for update in client_updates:
                    if 'masked_base' in update and param_name in update['masked_base']:
                        param_values.append(update['masked_base'][param_name])
                
                if param_values:
                    # Average the masked base parameters
                    stacked = torch.stack(param_values, dim=0)
                    agg_params['masked_base'][param_name] = torch.mean(stacked, dim=0)
        
        # Store aggregated base params for next round
        if 'masked_base' in agg_params:
            self.base_params = agg_params['masked_base']
        
        print(f"[FedWeIT Server] Aggregation complete")
        return agg_params 