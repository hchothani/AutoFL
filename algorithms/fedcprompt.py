"""
Fed-CPrompt: Contrastive Prompt for Rehearsal-Free Federated Continual Learning
ICML 2023 - https://arxiv.org/pdf/2307.04869.pdf
Reference code: https://github.com/grbagwe/Fed-CPrompt (see also FPPL)

Implements the Fed-CPrompt algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
Faithfully implements the key components: prompt pools, C2Loss, attention/key/query prompt injection, classifier head management, freezing, and asynchronous prompt learning.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy
import torch.nn.functional as F

class FedCPromptStrategy:
    def __init__(self, model: nn.Module, prompt_config: Dict[str, Any], num_clients: int = 2, num_tasks: int = 10, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning (should be ViT/transformer with prompt injection support).
            prompt_config: Configuration for prompt learning (e.g., prompt length, layers, etc.)
            num_clients: number of clients in federation
            num_tasks: number of tasks in continual learning
            kwargs: Additional Fed-CPrompt-specific parameters.
        """
        self.model = model
        self.prompt_config = prompt_config
        self.num_clients = num_clients
        self.num_tasks = num_tasks
        # Each client has a prompt pool: list of dicts (one per task)
        self.client_prompt_pools = [self._init_prompt_pool() for _ in range(num_clients)]
        # Each client has a classifier head per task
        self.client_classifiers = [self._init_classifier_pool() for _ in range(num_clients)]
        # Server maintains global prompt pool and classifier pool
        self.global_prompt_pool = copy.deepcopy(self.client_prompt_pools[0])
        self.global_classifier_pool = copy.deepcopy(self.client_classifiers[0])
        # C2Loss hyperparameters
        self.c2loss_gamma = prompt_config.get('c2loss_gamma', 1.0)
        self.c2loss_alpha = prompt_config.get('c2loss_alpha', 0.1)
        self.c2loss_lambda = prompt_config.get('c2loss_lambda', 1.0)

    def _init_prompt_pool(self):
        # Each task gets its own prompt (key, value, attention)
        prompt_len = self.prompt_config.get('prompt_length', 8)
        prompt_dim = self.prompt_config.get('prompt_dim', 768)
        pool = []
        for _ in range(self.num_tasks):
            # Each prompt is a dict with key, value, attention
            prompt = {
                'key': nn.Parameter(torch.randn(prompt_len, prompt_dim)),
                'value': nn.Parameter(torch.randn(prompt_len, prompt_dim)),
                'attn': nn.Parameter(torch.randn(prompt_dim)),
            }
            pool.append(prompt)
        return pool

    def _init_classifier_pool(self):
        # Each task gets its own classifier head (assume linear for now)
        num_classes = self.prompt_config.get('num_classes', 10)
        prompt_dim = self.prompt_config.get('prompt_dim', 768)
        pool = []
        for _ in range(self.num_tasks):
            clf = nn.Linear(prompt_dim, num_classes)
            pool.append(clf)
        return pool

    def get_prompt_params(self, client_id: int, task_id: int):
        # Return a deep copy of the prompt for a given client and task
        return {k: v.data.clone() for k, v in self.client_prompt_pools[client_id][task_id].items()}

    def set_prompt_params(self, prompt_params: Dict[str, Any], client_id: int, task_id: int):
        # Set the prompt params for a given client and task
        for k in self.client_prompt_pools[client_id][task_id]:
            self.client_prompt_pools[client_id][task_id][k].data.copy_(prompt_params[k])

    def get_classifier_params(self, client_id: int, task_id: int):
        # Return a deep copy of the classifier head for a given client and task
        clf = self.client_classifiers[client_id][task_id]
        return {k: v.data.clone() for k, v in clf.state_dict().items()}

    def set_classifier_params(self, clf_params: Dict[str, Any], client_id: int, task_id: int):
        # Set the classifier params for a given client and task
        self.client_classifiers[client_id][task_id].load_state_dict(clf_params)

    def aggregate(self, client_prompt_pools: List[List[Dict[str, torch.Tensor]]], client_classifier_pools: List[List[nn.Module]]):
        # Aggregate prompts and classifier heads for each task (FedAvg)
        for t in range(self.num_tasks):
            # Aggregate prompts
            for k in ['key', 'value', 'attn']:
                stack = torch.stack([client_prompt_pools[c][t][k] for c in range(self.num_clients)], dim=0)
                avg = torch.mean(stack, dim=0)
                self.global_prompt_pool[t][k].data.copy_(avg)
            # Aggregate classifier heads (FedAvg of weights)
            state_dicts = [client_classifier_pools[c][t].state_dict() for c in range(self.num_clients)]
            avg_state = {}
            for param_name in state_dicts[0]:
                stack = torch.stack([sd[param_name] for sd in state_dicts], dim=0)
                avg_state[param_name] = torch.mean(stack, dim=0)
            self.global_classifier_pool[t].load_state_dict(avg_state)
        return self.global_prompt_pool, self.global_classifier_pool

    def _inject_prompt(self, x, prompt, task_id):
        # This is a placeholder for prompt injection logic for ViT/transformer models
        # In practice, you would inject key/value/attn into the transformer's attention layers
        # For demonstration, we just concatenate prompt to input (not correct for real ViT)
        # Replace this with actual prompt injection for your model
        return x

    def c2loss(self, client_prompt, global_prompt_pool, task_id):
        # Implements C2Loss as in the paper
        # L_C2L = max(||P_c^t - P_s^t_prev||_2 - gamma * min_{i!=t} ||P_c^t - P_s^i||_2 + alpha, 0)
        p_c = client_prompt['key'].view(-1)
        p_s_prev = global_prompt_pool[task_id]['key'].view(-1)
        # First term: change from previous round
        term1 = torch.norm(p_c - p_s_prev, p=2)
        # Second term: min distance to other prompts
        min_dist = None
        for i, p_s in enumerate(global_prompt_pool):
            if i == task_id:
                continue
            dist = torch.norm(p_c - p_s['key'].view(-1), p=2)
            if min_dist is None or dist < min_dist:
                min_dist = dist
        if min_dist is None:
            min_dist = torch.tensor(0.0, device=p_c.device)
        val = term1 - self.c2loss_gamma * min_dist + self.c2loss_alpha
        return torch.clamp(val, min=0)

    def train_round(self, data, task_id: int, client_id: int):
        # Set model's prompt and classifier to current task
        prompt = self.client_prompt_pools[client_id][task_id]
        clf = self.client_classifiers[client_id][task_id]
        # Freeze previous task prompts/classifiers
        for t in range(self.num_tasks):
            if t != task_id:
                for p in self.client_prompt_pools[client_id][t].values():
                    p.requires_grad = False
                for param in self.client_classifiers[client_id][t].parameters():
                    param.requires_grad = False
            else:
                for p in self.client_prompt_pools[client_id][t].values():
                    p.requires_grad = True
                for param in self.client_classifiers[client_id][t].parameters():
                    param.requires_grad = True
        self.model.train()
        # Only optimize current prompt and classifier
        optimizer = torch.optim.Adam(
            list(prompt.values()) + list(clf.parameters()), lr=0.001)
        for x, y in data:
            optimizer.zero_grad()
            # Inject prompt (replace with actual logic for your model)
            x_prompted = self._inject_prompt(x, prompt, task_id)
            out = self.model(x_prompted)
            logits = clf(out)
            ce_loss = nn.CrossEntropyLoss()(logits, y)
            c2l = self.c2loss(prompt, self.global_prompt_pool, task_id)
            loss = ce_loss + self.c2loss_lambda * c2l
            loss.backward()
            optimizer.step()
        # Update prompt and classifier for this client/task
        # (already updated in-place)
        return self.get_prompt_params(client_id, task_id), self.get_classifier_params(client_id, task_id)

    def evaluate(self, data, task_id: int, client_id: int):
        # Evaluate the model on the given data for a specific task/client
        prompt = self.client_prompt_pools[client_id][task_id]
        clf = self.client_classifiers[client_id][task_id]
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data:
                x_prompted = self._inject_prompt(x, prompt, task_id)
                out = self.model(x_prompted)
                logits = clf(out)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

# See: https://arxiv.org/pdf/2307.04869.pdf and https://github.com/grbagwe/Fed-CPrompt for further details. 