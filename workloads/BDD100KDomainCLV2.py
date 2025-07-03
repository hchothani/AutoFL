import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset
from clutils.make_experiences import split_dataset
from typing import List, Tuple
import os
from PIL import Image
import json

from omegaconf import OmegaConf
from pathlib import Path

# Setup Config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config
cfg = load_config()

NUM_CLIENTS = cfg.server.num_clients
BATCH_SIZE = cfg.dataset.batch_size


class BDD100KDataset(Dataset):
    """BDD100K Dataset for domain incremental learning.
    
    This is a simplified version that assumes BDD100K images are organized
    in a specific folder structure. In practice, you would download and organize
    the actual BDD100K dataset.
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # In a real implementation, load actual BDD100K annotations
        # For now, creating a mock structure
        self.samples = []
        self.labels = []
        self.metadata = []
        
        # Simulate loading data with metadata
        # In reality, you would parse BDD100K JSON annotations
        self._load_mock_data()
    
    def _load_mock_data(self):
        """Mock data loading - replace with actual BDD100K loading logic"""
        # Simulate 1000 samples per split
        n_samples = 1000 if self.split == 'train' else 200
        
        # Define possible attributes
        weather_types = ['clear', 'cloudy', 'rainy', 'snowy', 'foggy']
        time_of_day = ['daytime', 'dawn/dusk', 'night']
        scenes = ['city street', 'highway', 'residential', 'parking lot']
        
        for i in range(n_samples):
            # Mock image path
            img_path = f"bdd100k_{self.split}_{i}.jpg"
            
            # Mock label (e.g., for segmentation or detection task)
            # For simplicity, using a classification label
            label = i % 10  # 10 classes
            
            # Mock metadata
            metadata = {
                'weather': np.random.choice(weather_types),
                'timeofday': np.random.choice(time_of_day),
                'scene': np.random.choice(scenes),
                'image_id': f"{self.split}_{i}"
            }
            
            self.samples.append(img_path)
            self.labels.append(label)
            self.metadata.append(metadata)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # In real implementation, load actual image
        # For now, create a dummy tensor
        img = torch.randn(3, 224, 224)  # Mock image
        
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        metadata = self.metadata[idx]
        
        return img, label, metadata


class TupleDataset(Dataset):
    """Convert dataset to tuple format for Avalanche"""
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = indices if indices is not None else list(range(len(dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label, _ = self.dataset[real_idx]
        return img, label


# Define domain-specific transforms
DOMAIN_TRANSFORMS = {
    'clear_day': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'rainy': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # Simulate rain with blur and reduced contrast
        transforms.GaussianBlur(kernel_size=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'night': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # Simulate night with reduced brightness
        transforms.ColorJitter(brightness=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'foggy': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # Simulate fog with strong blur and reduced contrast
        transforms.GaussianBlur(kernel_size=5),
        transforms.ColorJitter(contrast=0.5, saturation=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}


def create_domain_experiences(dataset, domain_conditions, transform, n_experiences=2):
    """Create experiences for a specific domain based on metadata conditions"""
    
    # Filter indices based on domain conditions
    domain_indices = []
    for idx in range(len(dataset)):
        _, _, metadata = dataset[idx]
        match = True
        for key, value in domain_conditions.items():
            if metadata.get(key) != value:
                match = False
                break
        if match:
            domain_indices.append(idx)
    
    if len(domain_indices) == 0:
        return []
    
    # Create subset with domain data
    domain_subset = TupleDataset(dataset, domain_indices)
    
    # Apply transform
    class TransformedDataset(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label
    
    transformed_dataset = TransformedDataset(domain_subset, transform)
    avalanche_dataset = AvalancheDataset(transformed_dataset)
    
    # Split into experiences
    experiences = split_dataset(avalanche_dataset, n_experiences=n_experiences)
    return experiences


def load_datasets(partition_id: int, data_dir: str = './data/bdd100k'):
    """Load BDD100K dataset with domain incremental experiences"""
    
    # Create base dataset
    train_dataset = BDD100KDataset(data_dir, split='train')
    test_dataset = BDD100KDataset(data_dir, split='test')
    
    # Simulate client data partitioning
    # In practice, use proper federated partitioning
    n_samples_per_client = len(train_dataset) // NUM_CLIENTS
    start_idx = partition_id * n_samples_per_client
    end_idx = start_idx + n_samples_per_client
    client_indices = list(range(start_idx, min(end_idx, len(train_dataset))))
    
    # Create client's subset
    client_train = Subset(train_dataset, client_indices)
    
    # Define domains based on BDD100K scenarios
    domains = [
        {'name': 'clear_day', 'conditions': {'weather': 'clear', 'timeofday': 'daytime'}},
        {'name': 'rainy', 'conditions': {'weather': 'rainy', 'timeofday': 'daytime'}},
        {'name': 'night', 'conditions': {'weather': 'clear', 'timeofday': 'night'}},
        {'name': 'foggy', 'conditions': {'weather': 'foggy', 'timeofday': 'daytime'}}
    ]
    
    # Create training experiences
    train_experiences = []
    for domain in domains:
        transform = DOMAIN_TRANSFORMS.get(domain['name'], DOMAIN_TRANSFORMS['clear_day'])
        experiences = create_domain_experiences(
            client_train, 
            domain['conditions'], 
            transform, 
            n_experiences=2
        )
        train_experiences.extend(experiences)
    
    # Create test experiences (one per domain)
    test_experiences = []
    for domain in domains:
        transform = DOMAIN_TRANSFORMS.get(domain['name'], DOMAIN_TRANSFORMS['clear_day'])
        test_exp = create_domain_experiences(
            test_dataset, 
            domain['conditions'], 
            transform, 
            n_experiences=1
        )
        if test_exp:
            test_experiences.extend(test_exp)
    
    # Create benchmark
    benchmark = benchmark_from_datasets(
        train_datasets=train_experiences,
        test_datasets=test_experiences
    )
    
    return benchmark


def get_dataloaders(partition_id: int):
    """Get data loaders for BDD100K domain incremental learning"""
    benchmark = load_datasets(partition_id)
    
    # Create dataloaders for each experience
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