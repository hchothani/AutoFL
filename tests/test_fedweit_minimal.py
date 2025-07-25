#!/usr/bin/env python3
"""
minimal test for fedweitstrategy with simplecnn and random data.
verifies local training, aggregation, and evaluation for two clients.
"""

import torch
import torch.nn as nn
import sys
import os

# add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.SimpleCNN import Net
from algorithms.fedweit import FedWeITStrategy

# config
num_clients = 2
num_classes = 10
input_shape = (3, 32, 32)
batch_size = 8
num_batches = 5
num_epochs = 2

# create random data loaders for each client
def make_random_loader():
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, *input_shape)
        y = torch.randint(0, num_classes, (batch_size,))
        data.append((x, y))
    return data

client_loaders = [make_random_loader() for _ in range(num_clients)]
test_loader = make_random_loader()

# create model and fedweit strategy
net = Net()
fedweit = FedWeITStrategy(net, sparsity=0.5, num_clients=num_clients, device='cpu')

print("="*50)
print("testing FedWeITStrategy minimal workflow")
print("="*50)

try:
    # local training for each client on a single task (task_id=0)
    client_updates = []
    for client_id in range(num_clients):
        print(f"training client {client_id} on task 0...")
        update = fedweit.train_task(client_loaders[client_id], task_id=0, client_id=client_id, epochs=num_epochs, lr=1e-2)
        client_updates.append(update)
    print("local training completed for all clients.")

    # aggregate at server
    print("aggregating updates at server...")
    fedweit.aggregate(client_updates)
    print("aggregation completed.")

    # evaluate global model for each client
    for client_id in range(num_clients):
        acc = fedweit.evaluate(test_loader, task_id=0, client_id=client_id)
        print(f"client {client_id} test accuracy on random data: {acc:.3f}")

    print("FedWeIT minimal test passed.")
except Exception as e:
    print(f"FedWeIT minimal test failed: {e}")
    import traceback
    traceback.print_exc() 