"""
FedMA: Federated Matched Averaging
NeurIPS 2020 - https://arxiv.org/abs/2002.06440
Reference code: https://github.com/IBM/FedMA

Implements the FedMA algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment

# hungarian matching utility for neuron/channel alignment
# cost_matrix: numpy array of shape (n, m)
# returns two lists of indices representing the optimal assignment
# see fedma paper section 2.1 and 2.2

def hungarian_matching(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

# match neurons (fc) or channels (conv) across clients using the hungarian algorithm
# returns a list of permuted weights for each client, aligned for averaging
# see fedma paper, algorithm 1

def match_layer_weights(client_weights: List[torch.Tensor], layer_type: str = 'fc') -> List[torch.Tensor]:
    # flatten neurons/channels for matching
    if layer_type == 'fc':
        # shape: (out_features, in_features)
        # match along out_features (neurons)
        features = [w.detach().cpu().numpy() for w in client_weights]
        n = features[0].shape[0]
    elif layer_type == 'conv':
        # shape: (out_channels, in_channels, k, k)
        # match along out_channels (channels)
        features = [w.detach().cpu().reshape(w.shape[0], -1).numpy() for w in client_weights]
        n = features[0].shape[0]
    else:
        raise ValueError('unsupported layer type')
    # use first client as reference for matching
    ref = features[0]
    permuted = [client_weights[0]]
    for i in range(1, len(features)):
        # compute cost matrix (euclidean distance between neurons/channels)
        cost = np.zeros((n, n))
        for j in range(n):
            for k in range(n):
                cost[j, k] = np.linalg.norm(ref[j] - features[i][k])
        row_ind, col_ind = hungarian_matching(cost)
        # permute client i's weights to align with reference
        if layer_type == 'fc':
            w_perm = client_weights[i][col_ind]
        else:  # conv
            w_perm = client_weights[i][col_ind]
        permuted.append(w_perm)
    return permuted

class FedMAStrategy:
    def __init__(self, model: nn.Module, ma_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        """
        implements the fedma algorithm for federated continual learning.
        see: https://arxiv.org/abs/2002.06440 and https://github.com/IBM/FedMA
        """
        self.model = model
        self.ma_config = ma_config
        self.num_clients = num_clients
        # store client models (deepcopy to avoid weight sharing)
        self.client_models = [copy.deepcopy(model) for _ in range(num_clients)]
        # store global model
        self.global_model = copy.deepcopy(model)

    def fedma_round(self, client_data: List[Any], num_layers: int = None):
        """
        perform a full fedma round: layer-wise matching, aggregation, freezing, and retraining.
        client_data: list of data loaders or iterables, one per client
        num_layers: number of layers to match (default: all trainable layers)
        follows fedma algorithm 1: match, aggregate, freeze, retrain, repeat
        """
        if num_layers is None:
            num_layers = len(list(self.model.state_dict().keys())) // 2  # rough guess: weight+bias per layer
        # get layer names (assume all clients have same structure)
        layer_names = [k for k in self.model.state_dict().keys() if 'weight' in k]
        for l, layer_name in enumerate(layer_names[:num_layers]):
            # 1. extract weights for this layer from all clients
            client_weights = [cm.state_dict()[layer_name].clone() for cm in self.client_models]
            # 2. match and permute using hungarian algorithm
            if len(client_weights[0].shape) == 2:
                layer_type = 'fc'
            elif len(client_weights[0].shape) == 4:
                layer_type = 'conv'
            else:
                continue  # skip unsupported
            permuted_weights = match_layer_weights(client_weights, layer_type)
            # 3. average matched weights to form global layer
            avg_weight = torch.stack(permuted_weights, dim=0).mean(dim=0)
            # 4. update global and client models with new global layer
            for i, cm in enumerate(self.client_models):
                sd = cm.state_dict()
                sd[layer_name] = avg_weight.clone()
                cm.load_state_dict(sd)
            self.global_model.state_dict()[layer_name].copy_(avg_weight)
            # 5. freeze this layer in all clients (no further updates)
            for cm in self.client_models:
                for name, param in cm.named_parameters():
                    if name == layer_name:
                        param.requires_grad = False
            # 6. retrain next layer(s) on each client (only unfrozen layers)
            if l+1 < num_layers:
                next_layer = layer_names[l+1]
                for client_id, cm in enumerate(self.client_models):
                    cm.train()
                    # only optimize next layer's params
                    optimizer = torch.optim.SGD([p for n, p in cm.named_parameters() if n == next_layer and p.requires_grad], lr=0.01)
                    for x, y in client_data[client_id]:
                        optimizer.zero_grad()
                        out = cm(x)
                        loss = nn.CrossEntropyLoss()(out, y)
                        loss.backward()
                        optimizer.step()
        # after all layers, update global model to match first client (all are now aligned)
        self.global_model.load_state_dict(self.client_models[0].state_dict())

    def aggregate(self, matched_models: List[nn.Module]):
        # aggregate matched client models (fedma logic)
        # perform layer-wise matching and averaging (see fedma paper, algorithm 1)
        state_dicts = [m.state_dict() for m in matched_models]
        global_state = copy.deepcopy(state_dicts[0])
        for k in global_state.keys():
            if 'weight' in k:
                ws = [sd[k] for sd in state_dicts]
                if len(ws[0].shape) == 2:
                    layer_type = 'fc'
                elif len(ws[0].shape) == 4:
                    layer_type = 'conv'
                else:
                    continue
                # match and permute for this layer
                ws_matched = match_layer_weights(ws, layer_type)
                # average matched weights
                global_state[k] = torch.stack(ws_matched, dim=0).mean(dim=0)
            elif 'bias' in k:
                # just average biases (no permutation needed)
                bs = [sd[k] for sd in state_dicts]
                global_state[k] = torch.stack(bs, dim=0).mean(dim=0)
        self.global_model.load_state_dict(global_state)
        return self.global_model

    def train_round(self, data, task_id: int, client_id: int):
        # perform a training round for a given task (FedMA-based local training)
        model = self.client_models[client_id]
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for x, y in data:
            optimizer.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            optimizer.step()
        self.client_models[client_id] = model
        return model

    def evaluate(self, data):
        # evaluate the global model on the given data
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data:
                out = self.global_model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0 