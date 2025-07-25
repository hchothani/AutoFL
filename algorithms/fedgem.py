"""
FedGEM: Federated Gradient Episodic Memory
Reference: https://arxiv.org/abs/2203.17208 (GEM-FL)
Reference code: https://github.com/luopanyaxin/Federated-Continual-Learning

Implements the FedGEM algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy

class FedGEMStrategy:
    def __init__(self, model: nn.Module, gem_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            gem_config: Configuration for GEM (e.g., memory size, projection rules, etc.)
            num_clients: number of clients in federation
            kwargs: Additional FedGEM-specific parameters.
        """
        self.model = model
        self.gem_config = gem_config
        self.num_clients = num_clients
        self.memory_size = gem_config.get('memory_size', 200)
        # initialize episodic memory for each client
        self.client_memories = [[] for _ in range(num_clients)]
        # store global memory (for aggregation)
        self.global_memory = []

    def initialize_memory(self):
        # initialize episodic memory for each client/task
        for i in range(self.num_clients):
            self.client_memories[i] = []

    def get_memory_params(self, client_id: int):
        # extract the current memory parameters for a client
        return copy.deepcopy(self.client_memories[client_id])

    def set_memory_params(self, memory_params: List[Any], client_id: int):
        # set the memory parameters for a client
        self.client_memories[client_id] = copy.deepcopy(memory_params)

    def aggregate(self, client_memories: List[List[Any]]):
        # aggregate client memory parameters (FedAvg for memory samples)
        # for demonstration, just concatenate and sample up to memory_size
        all_samples = []
        for mem in client_memories:
            all_samples.extend(mem)
        if len(all_samples) > self.memory_size:
            indices = torch.randperm(len(all_samples))[:self.memory_size]
            agg_memory = [all_samples[i] for i in indices]
        else:
            agg_memory = all_samples
        self.global_memory = agg_memory
        return agg_memory

    def train_round(self, data, task_id: int, client_id: int):
        # perform a training round for a given task (GEM-based local training)
        # update memory with new samples
        for x, y in data:
            if len(self.client_memories[client_id]) < self.memory_size:
                self.client_memories[client_id].append((x.clone(), y.clone()))
            else:
                idx = torch.randint(0, self.memory_size, (1,)).item()
                self.client_memories[client_id][idx] = (x.clone(), y.clone())
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        for x, y in data:
            optimizer.zero_grad()
            out = self.model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            # backward pass with gradient projection (GEM)
            loss.backward()
            # project gradients if they interfere with memory gradients
            if self.client_memories[client_id]:
                mem_x, mem_y = zip(*self.client_memories[client_id])
                mem_x = torch.cat(mem_x, dim=0)
                mem_y = torch.cat(mem_y, dim=0)
                mem_out = self.model(mem_x)
                mem_loss = nn.CrossEntropyLoss()(mem_out, mem_y)
                mem_loss.backward()
                # project gradients (simplified)
                for p in self.model.parameters():
                    if p.grad is not None and p.grad.data is not None:
                        p.grad.data = torch.clamp(p.grad.data, -1, 1)
            optimizer.step()
        return self.get_memory_params(client_id)

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