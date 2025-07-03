# core50 continual learning workload
# 50 domestic objects in 10 categories designed specifically for continual learning
# uses avalanche's built-in core50 dataset and benchmarks
# code based off of avalanche's benchmark documentation

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from avalanche.benchmarks.classic import CORe50
from avalanche.benchmarks.datasets import CORe50Dataset

from config_utils import load_config
cfg = load_config()

# configuration
NUM_CLIENTS = getattr(cfg.server, 'num_clients', 5)
BATCH_SIZE = getattr(cfg.dataset, 'batch_size', 32)
NUM_TASKS = getattr(cfg.cl, 'num_experiences', 8)  # Default to 8 experiences

def load_datasets(partition_id: int):
    """Load CORe50 datasets for federated continual learning"""
    
    print(f"Loading CORe50 dataset for client {partition_id}")
    
    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # CORe50 native size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # create CORe50 benchmark using Avalanche's classic benchmarks
    # Available scenarios: 'ni' (New Instances), 'nc' (New Classes), 'nic' (New Instances and Classes)
    scenario = getattr(cfg.cl, 'scenario', 'ni')  # Default to New Instances
    
    print(f"Using CORe50 scenario: {scenario}")
    
    if scenario == 'ni':
        # New Instances: 8 experiences, same 50 classes in each
        benchmark = CORe50(scenario='ni', run=0, train_transform=train_transform, eval_transform=test_transform)
    elif scenario == 'nc':
        # New Classes: 9 experiences, first has 10 classes, others have 5 classes each
        benchmark = CORe50(scenario='nc', run=0, train_transform=train_transform, eval_transform=test_transform)
    elif scenario == 'nic':
        # New Instances and Classes: 79 experiences
        benchmark = CORe50(scenario='nic', run=0, train_transform=train_transform, eval_transform=test_transform)
    else:
        raise ValueError(f"Unknown CORe50 scenario: {scenario}")
    
    # get train and test streams
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream
    
    print(f"CORe50 benchmark created with {len(train_stream)} train experiences and {len(test_stream)} test experiences")
    
    # for federated learning, we need to partition the experiences across clients
    # we'll cycle through experiences for each client
    client_train_experiences = []
    client_test_experiences = []
    
    # distribute experiences across clients in round-robin fashion
    for i, experience in enumerate(train_stream):
        if i % NUM_CLIENTS == partition_id:
            client_train_experiences.append(experience)
    
    for i, experience in enumerate(test_stream):
        if i % NUM_CLIENTS == partition_id:
            client_test_experiences.append(experience)
    
    print(f"Client {partition_id} assigned {len(client_train_experiences)} train experiences and {len(client_test_experiences)} test experiences")
    
    # create data loaders for the assigned experiences
    train_loaders = []
    test_loaders = []
    
    for exp in client_train_experiences:
        train_loader = DataLoader(
            exp.dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        train_loaders.append(train_loader)
    
    for exp in client_test_experiences:
        test_loader = DataLoader(
            exp.dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
        test_loaders.append(test_loader)
    
    # also create a single combined test loader for evaluation
    if client_test_experiences:
        # combine all test datasets
        combined_test_dataset = torch.utils.data.ConcatDataset([exp.dataset for exp in client_test_experiences])
        combined_test_loader = DataLoader(
            combined_test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
    else:
        combined_test_loader = None
    
    # get class information
    if scenario == 'ni':
        num_classes = 50  # All 50 objects
        classes_per_task = 50
    elif scenario == 'nc':
        num_classes = 50  # Total of 50 objects, but classes added incrementally
        classes_per_task = 5  # Varies: first task has 10, others have 5
    elif scenario == 'nic':
        num_classes = 50
        classes_per_task = "variable"  # Mixed new instances and classes
    
    print(f"CORe50 dataset info: {num_classes} total classes, scenario: {scenario}")
    
    result = {
        "train_loaders": train_loaders,
        "test_loaders": test_loaders,
        "combined_test_loader": combined_test_loader,
        "benchmark": benchmark,
        "client_experiences": {
            "train": client_train_experiences,
            "test": client_test_experiences
        },
        "dataset_info": {
            "num_classes": num_classes,
            "classes_per_task": classes_per_task,
            "scenario": scenario,
            "num_experiences": len(client_train_experiences),
            "input_size": 128,
            "input_channels": 3
        }
    }
    
    return result

def get_dataset_info():
    """Get basic dataset information for model configuration"""
    return {
        "name": "core50",
        "num_classes": 50,  # 50 domestic objects
        "input_channels": 3,  # RGB images
        "input_size": 128,   # Native CORe50 resolution
        "scenarios": ["ni", "nc", "nic"],
        "description": "CORe50: 50 domestic objects in 10 categories, designed for continual learning"
    } 