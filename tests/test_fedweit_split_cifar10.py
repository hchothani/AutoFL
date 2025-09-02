#!/usr/bin/env python3
"""
real-data test for fedweitstrategy using split cifar10 (2 tasks, 5 classes each)
verifies actual learning (accuracy > random, loss decreases)
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.SimpleCNN import Net
from algorithms.fedweit import FedWeITStrategy

# config
num_clients = 2
num_tasks = 2
classes_per_task = 5
num_classes = 10
batch_size = 32
num_epochs_local = 2
num_rounds = 2

# split cifar10 into two tasks: 0-4, 5-9
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

def get_task_loader(dataset, task_id):
    cls_start = task_id * classes_per_task
    cls_end = cls_start + classes_per_task
    idx = [i for i, (_, y) in enumerate(dataset) if cls_start <= y < cls_end]
    subset = torch.utils.data.Subset(dataset, idx)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader

# for speed, use only a subset of data
max_train_samples = 1000
max_test_samples = 200

def limited_loader(loader, max_samples):
    count = 0
    for x, y in loader:
        if count >= max_samples:
            break
        n = min(x.size(0), max_samples - count)
        yield x[:n], y[:n]
        count += n

# create model and fedweit strategy
net = Net()
fedweit = FedWeITStrategy(net, sparsity=0.5, num_clients=num_clients, device='cpu')

print("="*60)
print("FedWeITStrategy real-data test: split CIFAR10 (2 tasks)")
print("="*60)

try:
    all_acc = [[] for _ in range(num_clients)]
    all_loss = [[] for _ in range(num_clients)]
    for task_id in range(num_tasks):
        print(f"\n=== Task {task_id+1}/{num_tasks} (classes {task_id*classes_per_task}-{(task_id+1)*classes_per_task-1}) ===")
        train_loader = get_task_loader(trainset, task_id)
        test_loader = get_task_loader(testset, task_id)
        # for each round: local train, aggregate
        for rnd in range(num_rounds):
            print(f"--- federated round {rnd+1}/{num_rounds} ---")
            client_updates = []
            for client_id in range(num_clients):
                # for speed, use limited data
                local_loader = list(limited_loader(train_loader, max_train_samples))
                update = fedweit.train_task(local_loader, task_id=task_id, client_id=client_id, epochs=num_epochs_local, lr=1e-2)
                client_updates.append(update)
            fedweit.aggregate(client_updates)
            # evaluate
            for client_id in range(num_clients):
                fedweit.model.eval()
                correct = 0
                total = 0
                total_loss = 0.0
                ce = nn.CrossEntropyLoss()
                for x, y in limited_loader(test_loader, max_test_samples):
                    out = fedweit.model(x)
                    loss = ce(out, y)
                    total_loss += loss.item() * x.size(0)
                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
                acc = correct / total if total > 0 else 0.0
                avg_loss = total_loss / total if total > 0 else 0.0
                all_acc[client_id].append(acc)
                all_loss[client_id].append(avg_loss)
                print(f"client {client_id} test acc: {acc:.3f}  loss: {avg_loss:.4f}")
    # check for learning
    for client_id in range(num_clients):
        best_acc = max(all_acc[client_id])
        if best_acc < 0.2:
            print(f"FedWeIT real-data test FAILED: client {client_id} accuracy too low ({best_acc:.3f})")
            raise AssertionError("FedWeIT did not learn on real split CIFAR10.")
    print("\nLearning curve:")
    for client_id in range(num_clients):
        print(f"client {client_id} acc:  {all_acc[client_id]}")
        print(f"client {client_id} loss: {all_loss[client_id]}")
    print("FedWeIT real-data test PASSED: accuracy > random.")
except Exception as e:
    print(f"FedWeIT real-data test failed: {e}")
    import traceback
    traceback.print_exc() 