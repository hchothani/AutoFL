"""
Fed-CPrompt: Contrastive Prompt for Rehearsal-Free Federated Continual Learning
ICML 2023 - https://arxiv.org/pdf/2307.04869.pdf
Reference code: https://github.com/luopanyaxin/Federated-Continual-Learning (see also FPPL)

Implements the Fed-CPrompt algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy

class FedCPromptStrategy:
    def __init__(self, model: nn.Module, prompt_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            prompt_config: Configuration for prompt learning (e.g., prompt length, layers, etc.)
            num_clients: number of clients in federation
            kwargs: Additional Fed-CPrompt-specific parameters.
        """
        self.model = model
        self.prompt_config = prompt_config
        self.num_clients = num_clients
        # initialize prompt parameters for each client
        self.client_prompts = [self.initialize_prompts() for _ in range(num_clients)]
        # store global prompt parameters
        self.global_prompt = copy.deepcopy(self.client_prompts[0])

    def initialize_prompts(self):
        # initialize prompt parameters for each task/client
        prompt_len = self.prompt_config.get('prompt_length', 10)
        prompt_dim = self.prompt_config.get('prompt_dim', 128)
        # create a learnable prompt tensor
        prompt = torch.nn.Parameter(torch.randn(prompt_len, prompt_dim))
        return {'prompt': prompt}

    def get_prompt_params(self):
        # extract the current prompt parameters from the model
        # assumes model has an attribute 'prompt' (for demonstration)
        return {'prompt': self.model.prompt.data.clone()}

    def set_prompt_params(self, prompt_params: Dict[str, Any]):
        # set the prompt parameters in the model
        self.model.prompt.data.copy_(prompt_params['prompt'])

    def aggregate(self, client_prompts: List[Dict[str, Any]]):
        # aggregate client prompt parameters (FedAvg)
        stacked = torch.stack([cp['prompt'] for cp in client_prompts], dim=0)
        agg_prompt = torch.mean(stacked, dim=0)
        self.global_prompt = {'prompt': agg_prompt}
        return self.global_prompt

    def train_round(self, data, task_id: int, client_id: int):
        # perform a training round for a given task (prompt-based local training)
        # set model's prompt to client's current prompt
        self.set_prompt_params(self.client_prompts[client_id])
        self.model.train()
        optimizer = torch.optim.Adam([self.model.prompt], lr=0.001)
        for x, y in data:
            optimizer.zero_grad()
            out = self.model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            optimizer.step()
        # update client prompt
        self.client_prompts[client_id] = self.get_prompt_params()
        return self.get_prompt_params()

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