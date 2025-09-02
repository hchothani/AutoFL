"""
FedProto: Prototype-based Federated Continual Learning
ICLR 2022 - https://arxiv.org/abs/2106.02520
Reference code: https://github.com/luopanyaxin/Federated-Continual-Learning (see also HGP)

Implements the FedProto algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy

class FedProtoStrategy:
    def __init__(self, model: nn.Module, proto_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            proto_config: Configuration for prototype learning (e.g., prototype size, update rules, etc.)
            num_clients: number of clients in federation
            kwargs: Additional FedProto-specific parameters.
        """
        self.model = model
        self.proto_config = proto_config
        self.num_clients = num_clients
        self.lambda_proto = proto_config.get('lambda_proto', 1.0)
        # initialize prototypes for each client (dict: class_id -> (prototype tensor, count))
        self.client_protos = [{} for _ in range(num_clients)]
        # store global prototypes (dict: class_id -> prototype tensor)
        self.global_protos = {}

    def initialize_prototypes(self):
        # initialize prototype parameters for each class/client
        # for demonstration, do nothing (prototypes are built during training)
        pass

    def get_prototype_params(self, client_id: int):
        # extract the current prototype parameters for a client (deepcopy for safety)
        return copy.deepcopy(self.client_protos[client_id])

    def set_prototype_params(self, proto_params: Dict[str, Any], client_id: int):
        # set the prototype parameters for a client
        self.client_protos[client_id] = copy.deepcopy(proto_params)

    def aggregate(self, client_protos: List[Dict[int, tuple]]):
        # aggregate client prototype parameters (weighted average per class)
        all_classes = set()
        for cp in client_protos:
            all_classes.update(cp.keys())
        agg_protos = {}
        for cls in all_classes:
            weighted_sum = 0
            total_count = 0
            for cp in client_protos:
                if cls in cp:
                    proto, count = cp[cls]
                    weighted_sum += proto * count
                    total_count += count
            if total_count > 0:
                agg_protos[cls] = weighted_sum / total_count
        self.global_protos = agg_protos
        return agg_protos

    def train_round(self, data, task_id: int, client_id: int):
        # perform a training round for a given task (prototype-based local training)
        # build/update prototypes for each class in the batch
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        features = {}
        counts = {}
        # accumulate features and counts for each class
        for x, y in data:
            optimizer.zero_grad()
            out = self.model(x)
            ce_loss = nn.CrossEntropyLoss()(out, y)
            # prototype regularization loss
            proto_reg = 0.0
            with torch.no_grad():
                feats = self.model(x)
                for i, label in enumerate(y):
                    label = label.item()
                    if label not in features:
                        features[label] = feats[i].clone()
                        counts[label] = 1
                    else:
                        features[label] += feats[i].clone()
                        counts[label] += 1
            # after batch, compute batch prototypes
            batch_protos = {lbl: features[lbl] / counts[lbl] for lbl in features}
            # prototype regularization: sum squared distance to global prototype (if available)
            for lbl in batch_protos:
                if lbl in self.global_protos:
                    proto_reg += torch.norm(batch_protos[lbl] - self.global_protos[lbl]) ** 2
            loss = ce_loss + self.lambda_proto * proto_reg
            loss.backward()
            optimizer.step()
        # update prototypes as mean of features per class, and store counts
        for label in features:
            features[label] /= counts[label]
        # store as (prototype, count)
        self.client_protos[client_id] = {lbl: (features[lbl], counts[lbl]) for lbl in features}
        return self.get_prototype_params(client_id)

    def evaluate(self, data):
        # evaluate the model on the given data using nearest prototype
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data:
                feats = self.model(x)
                preds = []
                for i in range(feats.size(0)):
                    min_dist = float('inf')
                    pred = -1
                    for cls, proto in self.global_protos.items():
                        dist = torch.norm(feats[i] - proto)
                        if dist < min_dist:
                            min_dist = dist
                            pred = cls
                    preds.append(pred)
                preds = torch.tensor(preds, device=y.device)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0 