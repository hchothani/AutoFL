"""
FedRCIL: Federated Knowledge Distillation for Representation-based Contrastive Incremental Learning
ICCV Workshop 2023 - https://arxiv.org/abs/2308.09941
Reference code: (see paper and related repos)

Implements the FedRCIL algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy
import logging

# --- Exemplar Buffer for rehearsal ---
class ExemplarBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []  # list of (x, y)

    def add_examples(self, examples):
        self.buffer.extend(examples)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        idxs = torch.randperm(len(self.buffer))[:batch_size]
        return [self.buffer[i] for i in idxs]

# --- FedRCIL Strategy ---
class FedRCILStrategy:
    def __init__(self, model: nn.Module, rc_config: Dict[str, Any], num_clients: int = 2, buffer_size: int = 200, **kwargs):
        self.model = model
        self.rc_config = rc_config
        self.num_clients = num_clients
        self.buffer_size = buffer_size
        self.client_distills = [self.initialize_distillation() for _ in range(num_clients)]
        self.global_distill = copy.deepcopy(self.client_distills[0])
        self.client_buffers = [ExemplarBuffer(buffer_size) for _ in range(num_clients)]
        if not hasattr(self.model, 'projection'):
            proj_dim = self.rc_config.get('proj_dim', 128)
            self.model.projection = nn.Linear(self.model.fc.out_features, proj_dim)
        # task-incremental support
        self.current_task = 0
        self.task_class_map = kwargs.get('task_class_map', None)  # dict: task_id -> class list
        # logging
        self.logger = logging.getLogger('FedRCIL')
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            self.logger.addHandler(logging.StreamHandler())
        # checkpointing
        self.checkpoint_path = kwargs.get('checkpoint_path', None)

    def save_checkpoint(self, round_idx):
        if self.checkpoint_path:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'distill': self.global_distill,
                'round': round_idx
            }, f"{self.checkpoint_path}/fedrcil_round{round_idx}.pt")
            self.logger.info(f"Checkpoint saved at round {round_idx}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_distill = checkpoint['distill']
        self.logger.info(f"Checkpoint loaded from {path}")

    def update_task(self, task_id):
        self.current_task = task_id
        self.logger.info(f"Switched to task {task_id}")

    def initialize_distillation(self):
        temperature = self.rc_config.get('temperature', 0.5)
        proj_dim = self.rc_config.get('proj_dim', 128)
        projection = torch.nn.Parameter(torch.randn(proj_dim, proj_dim))
        return {'temperature': temperature, 'projection': projection}

    def get_distillation_params(self):
        return {'temperature': self.rc_config.get('temperature', 0.5), 'projection': self.model.projection.weight.data.clone()}

    def set_distillation_params(self, distill_params: Dict[str, Any]):
        self.model.projection.weight.data.copy_(distill_params['projection'])
        self.rc_config['temperature'] = distill_params['temperature']

    def aggregate(self, client_models: List[nn.Module], client_distills: List[Dict[str, Any]]):
        # FedAvg for model weights
        with torch.no_grad():
            for param in self.model.parameters():
                param.data.zero_()
            for client_model in client_models:
                for param, cparam in zip(self.model.parameters(), client_model.parameters()):
                    param.data.add_(cparam.data / len(client_models))
        # mean for temperature, FedAvg for projection
        temps = [cd['temperature'] for cd in client_distills]
        projs = [cd['projection'] for cd in client_distills]
        agg_temp = sum(temps) / len(temps)
        stacked = torch.stack(projs, dim=0)
        agg_proj = torch.mean(stacked, dim=0)
        self.global_distill = {'temperature': agg_temp, 'projection': agg_proj}
        return self.global_distill

    def train_round(self, data, task_id: int, client_id: int):
        """
        Perform a local training round for a given task and client.
        Includes rehearsal (buffer) loss and task-incremental support.
        """
        self.set_distillation_params(self.client_distills[client_id])
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        temperature = self.rc_config.get('temperature', 0.5)
        buffer = self.client_buffers[client_id]
        task_classes = None
        if self.task_class_map:
            task_classes = self.task_class_map.get(task_id, None)
        for x, y in data:
            optimizer.zero_grad()
            out = self.model(x)
            proj_out = self.model.projection(out)
            logits = proj_out / temperature
            labels = y
            # mask logits for task-incremental
            if task_classes is not None:
                mask = torch.zeros_like(logits)
                mask[:, task_classes] = 1
                logits = logits * mask
            loss = nn.CrossEntropyLoss()(logits, labels)
            # rehearsal loss (if buffer not empty)
            buf_samples = buffer.sample(len(x))
            if buf_samples:
                bx, by = zip(*buf_samples)
                bx = torch.stack(bx)
                by = torch.tensor(by)
                out_buf = self.model(bx)
                proj_buf = self.model.projection(out_buf)
                logits_buf = proj_buf / temperature
                if task_classes is not None:
                    mask = torch.zeros_like(logits_buf)
                    mask[:, task_classes] = 1
                    logits_buf = logits_buf * mask
                loss += nn.CrossEntropyLoss()(logits_buf, by)
            loss.backward()
            optimizer.step()
            # update buffer with new examples (FIFO)
            buffer.add_examples(list(zip(x, y)))
        self.client_distills[client_id] = self.get_distillation_params()
        self.logger.info(f"Client {client_id} finished training on task {task_id}")
        return self.get_distillation_params()

    def evaluate(self, data, task_id=None):
        """
        Evaluate the model on the given data. Optionally restrict to task classes.
        """
        self.model.eval()
        correct = 0
        total = 0
        task_classes = None
        if self.task_class_map and task_id is not None:
            task_classes = self.task_class_map.get(task_id, None)
        with torch.no_grad():
            for x, y in data:
                out = self.model(x)
                proj_out = self.model.projection(out)
                logits = proj_out
                if task_classes is not None:
                    mask = torch.zeros_like(logits)
                    mask[:, task_classes] = 1
                    logits = logits * mask
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total if total > 0 else 0.0
        self.logger.info(f"Evaluation accuracy: {acc:.4f}")
        return acc 