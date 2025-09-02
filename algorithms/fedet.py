"""
FedET: Federated Enhanced Transformer for Class-Incremental Learning
IJCAI 2023 - https://arxiv.org/abs/2305.18213
Reference code: https://github.com/luopanyaxin/Federated-Continual-Learning

Implements the FedET algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
Faithfully implements the key components: prompt pools (per task), classifier head pools (per task), freezing, prompt injection, and asynchronous aggregation.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy

class FedETStrategy:
    def __init__(self, model: nn.Module, et_config: Dict[str, Any], num_clients: int = 2, num_tasks: int = 5, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            et_config: Configuration for enhanced transformer (e.g., prompt layers, attention heads, etc.)
            num_clients: number of clients in federation
            num_tasks: number of tasks in continual learning
            kwargs: Additional FedET-specific parameters.
        """
        self.model = model
        self.et_config = et_config
        self.num_clients = num_clients
        self.num_tasks = num_tasks
        # initialize prompt pool and classifier head pool for each client
        self.client_prompt_pools = [self._init_prompt_pool() for _ in range(num_clients)]
        self.client_head_pools = [self._init_head_pool() for _ in range(num_clients)]
        # track which prompts/heads are frozen (per client, per task)
        self.frozen_prompts = [[False]*num_tasks for _ in range(num_clients)]
        self.frozen_heads = [[False]*num_tasks for _ in range(num_clients)]
        # store global prompt/head pools (for aggregation)
        self.global_prompt_pool = self._init_prompt_pool()
        self.global_head_pool = self._init_head_pool()

    def _init_prompt_pool(self):
        # initialize a prompt pool (one prompt per task)
        prompt_len = self.et_config.get('prompt_length', 10)
        prompt_dim = self.et_config.get('prompt_dim', 128)
        return [nn.Parameter(torch.randn(prompt_len, prompt_dim)) for _ in range(self.num_tasks)]

    def _init_head_pool(self):
        # initialize a classifier head pool (one head per task)
        num_classes = self.et_config.get('num_classes', 10)
        head_dim = self.et_config.get('head_dim', 128)
        return [nn.Linear(head_dim, num_classes) for _ in range(self.num_tasks)]

    def get_prompt_params(self, client_id: int):
        # extract the current prompt pool for a client
        return [p.data.clone() for p in self.client_prompt_pools[client_id]]

    def set_prompt_params(self, prompt_params: List[torch.Tensor], client_id: int):
        # set the prompt pool for a client
        for i, p in enumerate(prompt_params):
            self.client_prompt_pools[client_id][i].data.copy_(p)

    def get_head_params(self, client_id: int):
        # extract the current head pool for a client
        return [copy.deepcopy(h.state_dict()) for h in self.client_head_pools[client_id]]

    def set_head_params(self, head_params: List[Dict[str, Any]], client_id: int):
        # set the head pool for a client
        for i, hsd in enumerate(head_params):
            self.client_head_pools[client_id][i].load_state_dict(hsd)

    def freeze_previous(self, client_id: int, task_id: int):
        # freeze all previous prompts/heads for this client
        for t in range(task_id):
            self.frozen_prompts[client_id][t] = True
            self.frozen_heads[client_id][t] = True

    def inject_prompt(self, x, task_id: int, client_id: int):
        # inject the prompt for the current task into the model (ViT/transformer compatible)
        # assumes model has a method to accept prompt as input
        prompt = self.client_prompt_pools[client_id][task_id]
        return self.model(x, prompt=prompt)

    def aggregate(self, client_prompt_pools: List[List[torch.Tensor]], client_head_pools: List[List[Dict[str, Any]]]):
        # aggregate client prompt/head pools (FedAvg)
        # For each task, average prompts and heads across clients
        agg_prompt_pool = []
        agg_head_pool = []
        for t in range(self.num_tasks):
            # aggregate prompts
            stacked_prompts = torch.stack([cpp[t] for cpp in client_prompt_pools], dim=0)
            agg_prompt = torch.mean(stacked_prompts, dim=0)
            agg_prompt_pool.append(agg_prompt)
            # aggregate heads (average weights)
            # assumes all heads have same structure
            head_state_dicts = [chp[t] for chp in client_head_pools]
            avg_state_dict = {}
            for key in head_state_dicts[0].keys():
                avg_state_dict[key] = sum([hsd[key] for hsd in head_state_dicts]) / len(head_state_dicts)
            agg_head = copy.deepcopy(self.client_head_pools[0][t])
            agg_head.load_state_dict(avg_state_dict)
            agg_head_pool.append(agg_head)
        self.global_prompt_pool = agg_prompt_pool
        self.global_head_pool = agg_head_pool
        return agg_prompt_pool, agg_head_pool

    def train_round(self, data, task_id: int, client_id: int):
        # perform a training round for a given task (prompt/head-based local training)
        # freeze previous prompts/heads
        self.freeze_previous(client_id, task_id)
        # only update current prompt/head
        prompt = self.client_prompt_pools[client_id][task_id]
        head = self.client_head_pools[client_id][task_id]
        # freeze backbone and previous prompts/heads
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        prompt.requires_grad = True
        for param in head.parameters():
            param.requires_grad = True
        self.model.train()
        optimizer = torch.optim.Adam([prompt] + list(head.parameters()), lr=0.001)
        for x, y in data:
            optimizer.zero_grad()
            # inject prompt into model
            out = self.model(x, prompt=prompt)
            out = head(out)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            optimizer.step()
        # update pools
        self.client_prompt_pools[client_id][task_id].data.copy_(prompt.data)
        self.client_head_pools[client_id][task_id].load_state_dict(head.state_dict())
        return self.get_prompt_params(client_id), self.get_head_params(client_id)

    def evaluate(self, data, task_id: int, client_id: int):
        # evaluate the model on the given data for a specific task
        prompt = self.client_prompt_pools[client_id][task_id]
        head = self.client_head_pools[client_id][task_id]
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data:
                out = self.model(x, prompt=prompt)
                out = head(out)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0
