"""
PLoRA: Parameter-Efficient LoRA-based Federated Continual Learning
ICLR 2024 submission - https://arxiv.org/abs/2401.02094
Reference code: (see paper and related repos)

Implements the PLoRA algorithm for Federated Continual Learning (LoRA-based adaptation).
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy
import collections
import math

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.lora_a = nn.Parameter(torch.zeros((rank, base_layer.in_features)))
        self.lora_b = nn.Parameter(torch.zeros((base_layer.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        self.scaling = alpha / rank
        self.frozen = True
        for p in self.base.parameters():
            p.requires_grad = False
    def forward(self, x):
        base_out = self.base(x)
        lora_out = (self.lora_b @ (self.lora_a @ x.T)).T  # (batch, out)
        return base_out + self.scaling * lora_out

def inject_lora(model, rank=4, alpha=1.0, _prefix=''):
    """Recursively replace all nn.Linear layers with LoRALinear, only in nn.Module children. Debug print every replacement."""
    for name, module in model.named_children():
        full_name = f'{_prefix}.{name}' if _prefix else name
        if isinstance(module, nn.Linear):
            print(f'[LoRA Inject] Replacing {full_name} ({type(module)}) with LoRALinear')
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(model, name, lora_layer)
        elif isinstance(module, nn.Module):
            inject_lora(module, rank=rank, alpha=alpha, _prefix=full_name)
    return model

class PLoRAStrategy:
    def __init__(self, model: nn.Module, plora_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        self.rank = plora_config.get('rank', 4)
        self.alpha = plora_config.get('alpha', 1.0)
        self.model = inject_lora(model, rank=self.rank, alpha=self.alpha)
        print('[LoRA Inject] Model after injection:')
        print(self.model)
        self.plora_config = plora_config
        self.num_clients = num_clients
        # collect LoRA layer names
        self.lora_layers = [n for n, m in self.model.named_modules() if isinstance(m, LoRALinear)]
        # per-client LoRA params: {client: {layer: {'a':..., 'b':...}}}
        self.client_loras = [self._get_lora_params() for _ in range(num_clients)]
        self.global_lora = copy.deepcopy(self.client_loras[0])

    def _get_lora_params(self):
        params = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                params[name] = {
                    'a': module.lora_a.data.clone(),
                    'b': module.lora_b.data.clone()
                }
        return params

    def _set_lora_params(self, lora_params):
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear) and name in lora_params:
                module.lora_a.data.copy_(lora_params[name]['a'])
                module.lora_b.data.copy_(lora_params[name]['b'])

    def aggregate(self, client_loras: List[Dict[str, Any]]):
        # FedAvg per layer
        agg = {}
        for layer in self.lora_layers:
            a_stack = torch.stack([cl[layer]['a'] for cl in client_loras], dim=0)
            b_stack = torch.stack([cl[layer]['b'] for cl in client_loras], dim=0)
            agg[layer] = {
                'a': torch.mean(a_stack, dim=0),
                'b': torch.mean(b_stack, dim=0)
            }
        self.global_lora = agg
        return agg

    def train_round(self, data, task_id: int, client_id: int):
        self._set_lora_params(self.client_loras[client_id])
        self.model.train()
        lora_params = [p for n, m in self.model.named_modules() if isinstance(m, LoRALinear) for p in [m.lora_a, m.lora_b]]
        optimizer = torch.optim.Adam(lora_params, lr=0.001)
        for x, y in data:
            optimizer.zero_grad()
            out = self.model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            optimizer.step()
        self.client_loras[client_id] = self._get_lora_params()
        return self._get_lora_params()

    def evaluate(self, data):
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