# permuted mnist continual learning workload
# classic benchmark where different pixel permutations create different tasks


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
NUM_TASKS = getattr(cfg.cl, 'num_experiences', 5)

class PermutedMNISTDataset(Dataset):
    """dataset that applies permutation to mnist images"""
    
    def __init__(self, base_dataset, permutation=None):
        self.base_dataset = base_dataset
        self.permutation = permutation
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        if self.permutation is not None:
            # flatten image, apply permutation, reshape back
            image_flat = image.view(-1)
            image_permuted = image_flat[self.permutation]
            image = image_permuted.view(1, 28, 28)
        
        return image, label

def generate_permutations(num_tasks, seed=42):
    """generate random pixel permutations for each task"""
    np.random.seed(seed)
    permutations = []
    
    # first task: no permutation (identity)
    permutations.append(None)
    
    # subsequent tasks: random permutations
    for _ in range(num_tasks - 1):
        perm = np.random.permutation(28 * 28)
        permutations.append(torch.from_numpy(perm))
    
    return permutations

def load_datasets(partition_id: int):
    """load permuted mnist datasets for continual learning"""
    
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
    
    # generate permutations
    permutations = generate_permutations(NUM_TASKS)
    
    # create training experiences
    train_experiences = []
    for i, perm in enumerate(permutations):
        permuted_dataset = PermutedMNISTDataset(client_train, perm)
        aval_dataset = AvalancheDataset(permuted_dataset)
        train_experiences.append(aval_dataset)
        print(f"created train task {i}: {'identity' if perm is None else 'permuted'}")
    
    # create test experiences (use full test set for each task)
    test_experiences = []
    for i, perm in enumerate(permutations):
        permuted_test = PermutedMNISTDataset(test_dataset, perm)
        aval_test = AvalancheDataset(permuted_test)
        test_experiences.append(aval_test)
        print(f"created test task {i}: {'identity' if perm is None else 'permuted'}")
    
    # create benchmark
    benchmark = benchmark_from_datasets(
        train_datasets=train_experiences,
        test_datasets=test_experiences
    )
    
    print(f"created permuted mnist benchmark with {len(train_experiences)} tasks")
    print(f"client {partition_id}: {len(client_train)} training samples")
    
    return benchmark

def get_dataloaders(partition_id: int):
    """get data loaders for permuted mnist continual learning"""
    benchmark = load_datasets(partition_id)
    
    # create dataloaders for each experience
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