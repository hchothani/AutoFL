"""
STAMP: Spatio-Temporal Gradient Matching with Prototypical Coreset
arXiv 2025 - https://arxiv.org/abs/2506.12031
Reference code: (see paper and related repos)

Implements the STAMP algorithm for Federated Continual Learning.
This class is designed to be compatible with Avalanche/Flwr integration.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional
import copy
import collections

class STAMPStrategy:
    def __init__(self, model: nn.Module, stamp_config: Dict[str, Any], num_clients: int = 2, **kwargs):
        """
        Args:
            model: PyTorch model to be used for federated continual learning.
            stamp_config: Configuration for STAMP (e.g., coreset size, gradient matching rules, etc.)
            num_clients: number of clients in federation
            kwargs: Additional STAMP-specific parameters.
        """
        self.model = model
        self.stamp_config = stamp_config
        self.num_clients = num_clients
        self.coreset_size = stamp_config.get('coreset_size', 100)
        self.feature_layer = stamp_config.get('feature_layer', None)  # name of the layer to extract features from
        self.device = stamp_config.get('device', 'cpu')
        # initialize prototypical coreset for each client
        self.client_coresets = [self.initialize_coreset() for _ in range(num_clients)]
        # store global coreset
        self.global_coreset = copy.deepcopy(self.client_coresets[0])
        # store prototypes for each client and global
        self.client_prototypes = [{} for _ in range(num_clients)]  # Dict[class, prototype]
        self.global_prototypes = {}

    def initialize_coreset(self):
        # initialize prototypical coreset for each client/task
        return []

    def get_coreset_params(self, client_id: int):
        return copy.deepcopy(self.client_coresets[client_id])

    def set_coreset_params(self, coreset_params: list, client_id: int):
        self.client_coresets[client_id] = copy.deepcopy(coreset_params)

    def extract_features(self, x):
        # extract features from the model up to the specified layer
        if self.feature_layer is None:
            # fallback: flatten before classifier
            return x.view(x.size(0), -1)
        features = x
        module = self.model
        for name, layer in module.named_children():
            features = layer(features)
            if name == self.feature_layer:
                break
        return features

    def compute_prototypes(self, coreset):
        # compute mean feature vector (prototype) for each class in the coreset
        class_to_feats = collections.defaultdict(list)
        for x, y in coreset:
            with torch.no_grad():
                feats = self.extract_features(x.unsqueeze(0).to(self.device)).cpu()
            class_to_feats[y.item()].append(feats.squeeze(0))
        prototypes = {}
        for cls, feats in class_to_feats.items():
            prototypes[cls] = torch.stack(feats, dim=0).mean(dim=0)
        return prototypes

    def update_client_prototypes(self, client_id: int):
        self.client_prototypes[client_id] = self.compute_prototypes(self.client_coresets[client_id])

    def aggregate(self, client_coresets: List[list], client_prototypes: List[Dict[int, torch.Tensor]]):
        # aggregate client prototypes (spatial matching)
        # average prototypes for each class across clients
        all_classes = set()
        for proto in client_prototypes:
            all_classes.update(proto.keys())
        global_prototypes = {}
        for cls in all_classes:
            cls_protos = [proto[cls] for proto in client_prototypes if cls in proto]
            global_prototypes[cls] = torch.stack(cls_protos, dim=0).mean(dim=0)
        self.global_prototypes = global_prototypes
        # aggregate coreset samples as before (optional, for sample replay)
        all_samples = []
        for cs in client_coresets:
            all_samples.extend(cs)
        if len(all_samples) > self.coreset_size:
            indices = torch.randperm(len(all_samples))[:self.coreset_size]
            agg_coreset = [all_samples[i] for i in indices]
        else:
            agg_coreset = all_samples
        self.global_coreset = agg_coreset
        return agg_coreset, global_prototypes

    def prototype_loss(self, features, targets, prototypes):
        # encourage features to be close to their class prototype (temporal matching)
        loss = 0.0
        for i in range(features.size(0)):
            cls = targets[i].item()
            if cls in prototypes:
                loss += torch.nn.functional.mse_loss(features[i], prototypes[cls])
        return loss / features.size(0)

    def train_round(self, data, task_id: int, client_id: int, lambda_proto: float = 1.0):
        # update coreset with new samples
        # data is now a tuple (xs, ys) where xs: [batch, 1, 28, 28], ys: [batch]
        xs, ys = data
        for i in range(xs.size(0)):
            x, y = xs[i], ys[i]
            if len(self.client_coresets[client_id]) < self.coreset_size:
                self.client_coresets[client_id].append((x.clone(), y.clone()))
            else:
                idx = torch.randint(0, self.coreset_size, (1,)).item()
                self.client_coresets[client_id][idx] = (x.clone(), y.clone())
        # update client prototypes
        self.update_client_prototypes(client_id)
        # local training with prototype regularization (temporal matching)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        xs, ys = xs.to(self.device), ys.to(self.device)
        optimizer.zero_grad()
        out = self.model(xs)
        ce_loss = nn.CrossEntropyLoss()(out, ys)
        feats = self.extract_features(xs)
        proto_loss = self.prototype_loss(feats, ys, self.client_prototypes[client_id])
        loss = ce_loss + lambda_proto * proto_loss
        loss.backward()
        optimizer.step()
        return self.get_coreset_params(client_id), self.client_prototypes[client_id]

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