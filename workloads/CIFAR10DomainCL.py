import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import flwr
from flwr_datasets import FederatedDataset
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset, make_avalanche_dataset
from clutils.make_experiences import split_dataset

from omegaconf import OmegaConf
from pathlib import Path

# Setup Config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config
cfg = load_config()

NUM_CLIENTS = cfg.server.num_clients
BATCH_SIZE = cfg.dataset.batch_size

class TupleDataset(torch.utils.data.Dataset):
    """Convert HuggingFace dataset format to tuple format for Avalanche"""
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample["img"], sample["label"]

# Define different domain transformations
DOMAIN_TRANSFORMS = {
    'original': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'brightness': transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'contrast': transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(contrast=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'blur': transforms.Compose([
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Create experiences for each domain
    train_experiences = []
    for domain_name, transform in DOMAIN_TRANSFORMS.items():
        # Apply domain-specific transform with proper function
        def apply_transforms_closure(batch, t=transform):
            batch["img"] = [t(img) for img in batch["img"]]
            return batch
        
        domain_data = partition_train_test["train"].with_transform(apply_transforms_closure)
        # Convert to tuple format for Avalanche compatibility
        domain_data_tuple = TupleDataset(domain_data)
        # Convert to AvalancheDataset
        domain_data_av = AvalancheDataset(domain_data_tuple)
        # Split into experiences
        domain_experiences = split_dataset(domain_data_av, n_experiences=2)  # Split each domain into 2 experiences
        train_experiences.extend(domain_experiences)
    
    # Create test experiences
    test_experiences = []
    testset = fds.load_split("test")
    for domain_name, transform in DOMAIN_TRANSFORMS.items():
        def apply_transforms_closure_test(batch, t=transform):
            batch["img"] = [t(img) for img in batch["img"]]
            return batch
        
        domain_test = testset.with_transform(apply_transforms_closure_test)
        # Convert to tuple format for Avalanche compatibility
        domain_test_tuple = TupleDataset(domain_test)
        # Convert to AvalancheDataset
        domain_test_av = AvalancheDataset(domain_test_tuple)
        test_experiences.append(domain_test_av)
    
    # Create benchmark
    benchmark = benchmark_from_datasets(
        train_datasets=train_experiences,
        test_datasets=test_experiences
    )
    
    return benchmark

def get_dataloaders(partition_id: int):
    benchmark = load_datasets(partition_id)
    
    # Create dataloaders for each experience
    train_loaders = []
    for exp in benchmark.train_stream:
        train_loaders.append(
            DataLoader(exp.dataset, batch_size=BATCH_SIZE, shuffle=True)
        )
    
    test_loaders = []
    for exp in benchmark.test_stream:
        test_loaders.append(
            DataLoader(exp.dataset, batch_size=BATCH_SIZE)
        )
    
    return train_loaders, test_loaders 