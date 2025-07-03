# basic mnist workload for federated learning
# standard mnist classification without continual learning transformations

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.datasets as datasets
import numpy as np
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets

from config_utils import load_config
cfg = load_config()

# configuration
NUM_CLIENTS = getattr(cfg.server, 'num_clients', 5)
BATCH_SIZE = getattr(cfg.dataset, 'batch_size', 32)
NUM_TASKS = getattr(cfg.cl, 'num_experiences', 1)  # Single task for basic MNIST

def load_datasets(partition_id: int):
    """load basic mnist datasets for federated learning"""
    
    # base transforms
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # load base mnist dataset
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=base_transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=base_transform
    )
    
    # create client data partition
    n_samples_per_client = len(train_dataset) // NUM_CLIENTS
    start_idx = partition_id * n_samples_per_client
    end_idx = start_idx + n_samples_per_client
    client_indices = list(range(start_idx, min(end_idx, len(train_dataset))))
    
    client_train = Subset(train_dataset, client_indices)
    
    # create single experience for basic classification
    train_experiences = []
    aval_dataset = AvalancheDataset(client_train)
    train_experiences.append(aval_dataset)
    print(f"created basic mnist train experience with {len(client_train)} samples")
    
    # create test experience (use full test set)
    test_experiences = []
    aval_test = AvalancheDataset(test_dataset)
    test_experiences.append(aval_test)
    print(f"created basic mnist test experience with {len(test_dataset)} samples")
    
    # create benchmark
    benchmark = benchmark_from_datasets(
        train_datasets=train_experiences,
        test_datasets=test_experiences
    )
    
    print(f"created basic mnist benchmark with {len(train_experiences)} task")
    print(f"client {partition_id}: {len(client_train)} training samples")
    
    return benchmark

def get_dataloaders(partition_id: int):
    """get data loaders for basic mnist"""
    benchmark = load_datasets(partition_id)
    
    # create dataloaders for the single experience
    train_loaders = []
    for exp in benchmark.train_stream:
        train_loaders.append(
            DataLoader(exp.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        )
    
    test_loaders = []
    for exp in benchmark.test_stream:
        test_loaders.append(
            DataLoader(exp.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        )
    
    return train_loaders, test_loaders 