# Quick Start Guide for Domain Incremental Workloads


# THIS IS OLD ignore it for now


This guide will help you quickly get started with the domain incremental workloads for Continual Federated Learning.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vimarsh244/AutoFL
cd AutoFL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Your First Experiment

### 1. Choose a Workload

Select one of the available workloads:
- CIFAR10DomainCL
- CIFAR100DomainCL
- BDD100KDomainCL
- KITTIDomainCL

### 2. Basic Example

Here's a minimal example using CIFAR10:

```python
from workloads.CIFAR10DomainCL import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim

# Get dataloaders for client 0
train_loaders, test_loaders = get_dataloaders(partition_id=0)

# Define a simple model
model = nn.Sequential(
    nn.Conv2d(3, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 15 * 15, 10)
)

# Training loop for one experience
def train_experience(model, train_loader, epochs=5):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch in train_loader:
            images, labels = batch['img'], batch['label']
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Train on first experience
train_experience(model, train_loaders[0])
```

### 3. Federated Learning Setup

For federated learning experiments:

1. Start the server:
```bash
python mclserver.py
```

2. Start clients (in separate terminals):
```bash
python mclclient.py --partition-id 0
python mclclient.py --partition-id 1
# ... start more clients as needed
```

## Configuration

1. Adjust client settings in `config/config.yaml`:
```yaml
server:
  num_rounds: 10
  num_clients: 10
  fraction_fit: 0.8
  fraction_eval: 0.2
  min_fit: 8
  min_eval: 2
```

2. Modify workload parameters in the respective workload file:
```python
NUM_CLIENTS = 10  # Number of federated clients
BATCH_SIZE = 32   # Batch size for training
```

## Common Tasks

### 1. Adding a New Domain

For CIFAR workloads:
```python
DOMAIN_TRANSFORMS['new_domain'] = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(hue=0.5),  # Example: hue variation
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

For BDD100K/KITTI:
```python
domains['new_domain'] = {
    'weather': 'new_weather',
    'timeofday': 'new_time'
}
```

### 2. Changing Number of Experiences

Modify the `split_dataset` call in the workload file:
```python
domain_experiences = split_dataset(domain_data, n_experiences=3)  # Change from 2 to 3
```

### 3. Monitoring Training

Add logging to track performance:
```python
import wandb

wandb.init(project="domain-incremental-fl")
wandb.log({
    "loss": loss.item(),
    "accuracy": accuracy,
    "domain": domain_name
})
```