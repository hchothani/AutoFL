"""
SacFL: Self-Adaptive Federated Continual Learning for Resource-Constrained End Devices
TNNLS 2025 - https://arxiv.org/abs/2505.00365
Implements the SacFL algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.

Key features:
- Encoder (task-robust/global) and Decoder (task-sensitive/local) split
- Contrastive learning for representation and shift detection
- Autonomous shift detection and self-adaptive logic
- Only encoder is aggregated globally; decoder remains local
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy

class SacFLEncoderDecoder(nn.Module):
    """
    A simple encoder-decoder wrapper for SacFL. The encoder is the global (task-robust) part,
    the decoder is the local (task-sensitive) part. This structure is required for proper aggregation.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

class SacFLStrategy:
    """
    Implements the SacFL algorithm as described in the paper:
    'SacFL: Self-Adaptive Federated Continual Learning for Resource-Constrained End Devices'
    https://arxiv.org/abs/2505.00365
    """
    def __init__(self, model: SacFLEncoderDecoder, sacfl_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        self.model = model
        self.sacfl_config = sacfl_config
        self.num_clients = num_clients
        self.client_decoders = [copy.deepcopy(model.decoder) for _ in range(num_clients)]
        self.global_encoder = copy.deepcopy(model.encoder)
        self.shift_threshold = sacfl_config.get('shift_threshold', 1.0)
        self.last_contrastive_loss = [0.0 for _ in range(num_clients)]
        # Simple memory buffer for contrastive learning (per client)
        self.memory_size = sacfl_config.get('memory_size', 128)
        self.memory_buffers = [[] for _ in range(num_clients)]  # list of lists of tensors
        self.temperature = sacfl_config.get('contrastive_temperature', 0.5)

    def aggregate_encoder(self, client_encoders: List[nn.Module]):
        # Aggregate encoder parameters (FedAvg)
        state_dicts = [enc.state_dict() for enc in client_encoders]
        agg_state = copy.deepcopy(state_dicts[0])
        for k in agg_state:
            agg_state[k] = sum([sd[k] for sd in state_dicts]) / len(state_dicts)
        self.global_encoder.load_state_dict(agg_state)
        return self.global_encoder

    def set_client_decoder(self, client_id: int):
        self.model.decoder = self.client_decoders[client_id]

    def set_global_encoder(self):
        self.model.encoder = self.global_encoder

    def train_round(self, data, task_id: int, client_id: int):
        self.set_global_encoder()
        self.set_client_decoder(client_id)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for x, y in data:
            optimizer.zero_grad()
            out = self.model(x)
            ce_loss = nn.CrossEntropyLoss()(out, y)
            with torch.no_grad():
                features = self.model.encoder(x)
                self._update_memory(client_id, features)
            contrastive_loss = self.compute_contrastive_loss(client_id, x)
            total_loss = ce_loss + self.sacfl_config.get('contrastive_weight', 1.0) * contrastive_loss
            total_loss.backward()
            optimizer.step()
        self.client_decoders[client_id] = copy.deepcopy(self.model.decoder)
        self.last_contrastive_loss[client_id] = contrastive_loss.item()
        # Self-adaptive logic: if shift detected, trigger defense (reset decoder)
        if self.detect_shift(client_id):
            # In the SacFL paper, the device can autonomously trigger CL or defense
            # Here, as a simple defense, we reset the decoder for this client
            self.client_decoders[client_id].apply(self._reset_weights)
        return copy.deepcopy(self.model.encoder), copy.deepcopy(self.model.decoder)

    def _reset_weights(self, m):
        """
        Utility to reset module weights (for defense/continual learning trigger).
        """
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    def _update_memory(self, client_id: int, features: torch.Tensor):
        # Add features to the memory buffer (FIFO)
        buffer = self.memory_buffers[client_id]
        for f in features:
            buffer.append(f.detach().cpu())
        if len(buffer) > self.memory_size:
            self.memory_buffers[client_id] = buffer[-self.memory_size:]

    def compute_contrastive_loss(self, client_id: int, x):
        """
        Compute NT-Xent (SimCLR-style) contrastive loss using the memory buffer.
        This is a simplified version for illustration; in practice, use augmentations.
        """
        device = x.device
        encoder = self.model.encoder
        features = encoder(x)
        features = nn.functional.normalize(features, dim=1)
        # Gather memory buffer
        mem = self.memory_buffers[client_id]
        if len(mem) < 2:
            return torch.tensor(0.0, device=device)
        mem_feats = torch.stack(mem).to(device)
        mem_feats = nn.functional.normalize(mem_feats, dim=1)
        # For each feature in batch, treat as anchor, rest as negatives
        loss = 0.0
        n = features.size(0)
        for i in range(n):
            anchor = features[i]
            # Positive: nearest in memory (simulate augmentation)
            pos_idx = torch.randint(0, mem_feats.size(0), (1,)).item()
            positive = mem_feats[pos_idx]
            # Negatives: all others in memory
            negatives = torch.cat([mem_feats[:pos_idx], mem_feats[pos_idx+1:]], dim=0) if mem_feats.size(0) > 1 else mem_feats
            pos_sim = torch.exp(torch.dot(anchor, positive) / self.temperature)
            neg_sim = torch.exp(torch.matmul(anchor, negatives.t()) / self.temperature).sum()
            loss += -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
        return loss / n

    def detect_shift(self, client_id: int):
        # Detect task/data shift using contrastive loss statistics
        return self.last_contrastive_loss[client_id] > self.shift_threshold

    def evaluate(self, data, client_id: int):
        self.set_global_encoder()
        self.set_client_decoder(client_id)
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