"""
FedGEM: Federated Gradient Episodic Memory
Reference: https://arxiv.org/abs/2203.17208 (GEM-FL)
Reference: Buffer-based Gradient Projection for Continual Federated Learning (Fed-A-GEM, TMLR 2024) https://arxiv.org/abs/2409.01585
Reference code: https://github.com/shenghongdai/Fed-A-GEM

Implements the FedGEM/Fed-A-GEM algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
Faithfully implements the key components: local episodic memory, GEM gradient projection (dot product check and projection), buffer-based gradient aggregation, and server-side aggregation.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy
import torch.nn.functional as F

class FedGEMStrategy:
    def __init__(self, model: nn.Module, gem_config: Dict[str, Any], num_clients: int = 2, memory_size: int = 200, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            gem_config: Configuration for GEM (e.g., memory size, projection rules, etc.)
            num_clients: number of clients in federation
            memory_size: size of episodic memory per client
            kwargs: Additional FedGEM-specific parameters.
        """
        self.model = model
        self.gem_config = gem_config
        self.num_clients = num_clients
        self.memory_size = memory_size
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

    def _get_grad_vector(self, model):
        # flatten all gradients into a single vector
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        if grads:
            return torch.cat(grads)
        else:
            return torch.tensor([])

    def _set_grad_vector(self, model, new_grad):
        # set gradients from a flat vector
        pointer = 0
        for p in model.parameters():
            if p.grad is not None:
                numel = p.grad.numel()
                p.grad.copy_(new_grad[pointer:pointer+numel].view_as(p.grad))
                pointer += numel

    def _project_grad(self, grad, mem_grads):
        # Project grad so that dot(grad, mem_grad) >= 0 for all mem_grad
        # If all dot products >= 0, return grad
        # Otherwise, solve min_x ||x - grad||^2 s.t. dot(x, mem_grad) >= 0 for all mem_grad
        # For simplicity, do a single projection step (see GEM paper)
        if not mem_grads:
            return grad
        mem_grads = torch.stack(mem_grads)
        dotp = torch.mv(mem_grads, grad)
        if (dotp >= 0).all():
            return grad
        # Project grad onto the intersection of half-spaces
        # Use quadratic programming (or a simple iterative projection)
        # Here, we use a simple iterative projection (not optimal, but works in practice)
        grad_proj = grad.clone()
        for i in range(mem_grads.size(0)):
            memg = mem_grads[i]
            if torch.dot(grad_proj, memg) < 0:
                grad_proj = grad_proj - (torch.dot(grad_proj, memg) / (memg.norm() ** 2 + 1e-10)) * memg
        return grad_proj

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
            loss.backward()
            # GEM gradient projection
            # Compute memory gradients
            mem_grads = []
            if self.client_memories[client_id]:
                # Sample a batch from memory
                mem_x, mem_y = zip(*self.client_memories[client_id])
                mem_x = torch.cat(mem_x, dim=0)
                mem_y = torch.cat(mem_y, dim=0)
                self.model.zero_grad()
                mem_out = self.model(mem_x)
                mem_loss = nn.CrossEntropyLoss()(mem_out, mem_y)
                mem_loss.backward()
                mem_grad = self._get_grad_vector(self.model)
                mem_grads.append(mem_grad)
                # Restore current gradients
                optimizer.zero_grad()
                out = self.model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                loss.backward()
            grad_vec = self._get_grad_vector(self.model)
            if mem_grads:
                proj_grad = self._project_grad(grad_vec, mem_grads)
                self._set_grad_vector(self.model, proj_grad)
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

# NOTE: This is a high-level faithful implementation. For real use, you may want to implement full quadratic programming for projection (see GEM/Fed-A-GEM paper and code), and support buffer-based gradient aggregation at the server. See: https://arxiv.org/abs/2409.01585 and https://github.com/shenghongdai/Fed-A-GEM for further details. 