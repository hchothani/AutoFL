import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset
from clutils.make_experiences import split_dataset
from typing import List, Tuple, Dict
import os
from PIL import Image

from omegaconf import OmegaConf
from pathlib import Path

# Setup Config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config
cfg = load_config()

NUM_CLIENTS = cfg.server.num_clients
BATCH_SIZE = cfg.dataset.batch_size


class KITTIDataset(Dataset):
    """KITTI Dataset for domain incremental learning.
    
    This is a simplified version for autonomous driving scenarios.
    In practice, you would load actual KITTI dataset with proper annotations.
    """
    
    def __init__(self, root_dir, split='train', task='object_detection', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.task = task
        self.transform = transform
        
        # Mock data structure
        self.samples = []
        self.labels = []
        self.metadata = []
        
        # Load mock data
        self._load_mock_data()
    
    def _load_mock_data(self):
        """Mock data loading - replace with actual KITTI loading logic"""
        # Simulate samples
        n_samples = 800 if self.split == 'train' else 200
        
        # Define KITTI-specific attributes
        weather_conditions = ['sunny', 'cloudy', 'overcast']
        road_types = ['urban', 'highway', 'residential', 'campus']
        traffic_density = ['low', 'medium', 'high']
        time_periods = ['morning', 'noon', 'afternoon']
        
        # Object classes for KITTI
        object_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist', 'Tram', 'Misc', 'DontCare']
        
        for i in range(n_samples):
            # Mock image path
            img_path = f"kitti_{self.split}_{i:06d}.png"
            
            # Mock label based on task
            if self.task == 'object_detection':
                # Simulate bounding boxes and classes
                n_objects = np.random.randint(1, 10)
                label = {
                    'boxes': torch.rand(n_objects, 4),  # Mock bounding boxes
                    'labels': torch.randint(0, len(object_classes), (n_objects,)),
                    'num_objects': n_objects
                }
            else:  # classification task
                label = i % len(object_classes)
            
            # Mock metadata
            metadata = {
                'weather': np.random.choice(weather_conditions),
                'road_type': np.random.choice(road_types),
                'traffic': np.random.choice(traffic_density),
                'time_period': np.random.choice(time_periods),
                'sequence_id': f"seq_{i//10:04d}",
                'frame_id': i % 10
            }
            
            self.samples.append(img_path)
            self.labels.append(label)
            self.metadata.append(metadata)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # In real implementation, load actual image
        # For now, create a dummy tensor (KITTI images are typically larger)
        img = torch.randn(3, 375, 1242)  # KITTI typical resolution
        
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        metadata = self.metadata[idx]
        
        return img, label, metadata


class SimplifiedKITTIDataset(Dataset):
    """Simplified KITTI dataset for classification tasks"""
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = indices if indices is not None else list(range(len(dataset)))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label, _ = self.dataset[real_idx]
        
        # For object detection, convert to classification by counting objects
        if isinstance(label, dict):
            # Use number of cars as classification label (simplified)
            car_label = 0  # Car class in KITTI
            n_cars = (label['labels'] == car_label).sum().item()
            label = min(n_cars, 9)  # Cap at 9 for 10-class classification
        
        return img, label


# Define domain-specific transforms for KITTI
DOMAIN_TRANSFORMS = {
    'urban_sunny': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for efficiency
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'highway': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.3),  # Less flipping for highway
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Motion blur
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'residential': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),  # Slight rotation for turns
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'challenging': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        # Simulate challenging conditions
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.GaussianBlur(kernel_size=5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}


def create_kitti_domain_experiences(dataset, domain_conditions, transform, n_experiences=2):
    """Create experiences for KITTI based on driving scenarios"""
    
    # Filter indices based on domain conditions
    domain_indices = []
    for idx in range(len(dataset)):
        _, _, metadata = dataset[idx]
        match = True
        for key, value in domain_conditions.items():
            if isinstance(value, list):
                if metadata.get(key) not in value:
                    match = False
                    break
            else:
                if metadata.get(key) != value:
                    match = False
                    break
        if match:
            domain_indices.append(idx)
    
    if len(domain_indices) == 0:
        return []
    
    # Create subset with domain data
    domain_subset = SimplifiedKITTIDataset(dataset, domain_indices)
    
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


def load_datasets(partition_id: int, data_dir: str = './data/kitti'):
    """Load KITTI dataset with domain incremental experiences"""
    
    # Create base dataset
    train_dataset = KITTIDataset(data_dir, split='train')
    test_dataset = KITTIDataset(data_dir, split='test')
    
    # Simulate client data partitioning
    n_samples_per_client = len(train_dataset) // NUM_CLIENTS
    start_idx = partition_id * n_samples_per_client
    end_idx = start_idx + n_samples_per_client
    client_indices = list(range(start_idx, min(end_idx, len(train_dataset))))
    
    # Create client's subset
    client_train = Subset(train_dataset, client_indices)
    
    # Define domains based on KITTI driving scenarios
    domains = [
        {
            'name': 'urban_sunny',
            'conditions': {
                'road_type': 'urban',
                'weather': 'sunny'
            }
        },
        {
            'name': 'highway',
            'conditions': {
                'road_type': 'highway',
                'weather': ['sunny', 'cloudy']
            }
        },
        {
            'name': 'residential',
            'conditions': {
                'road_type': 'residential',
                'traffic': ['low', 'medium']
            }
        },
        {
            'name': 'challenging',
            'conditions': {
                'weather': ['cloudy', 'overcast'],
                'traffic': 'high'
            }
        }
    ]
    
    # Create training experiences
    train_experiences = []
    for domain in domains:
        transform = DOMAIN_TRANSFORMS.get(domain['name'], DOMAIN_TRANSFORMS['urban_sunny'])
        experiences = create_kitti_domain_experiences(
            client_train, 
            domain['conditions'], 
            transform, 
            n_experiences=2
        )
        train_experiences.extend(experiences)
    
    # Create test experiences (one per domain)
    test_experiences = []
    for domain in domains:
        transform = DOMAIN_TRANSFORMS.get(domain['name'], DOMAIN_TRANSFORMS['urban_sunny'])
        test_exp = create_kitti_domain_experiences(
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
    """Get data loaders for KITTI domain incremental learning"""
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