# rotated mnist continual learning workload
# classic benchmark where mnist digits are rotated at different angles to create different tasks

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.datasets as datasets
import numpy as np
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from PIL import Image
import torchvision.transforms.functional as TF

from config_utils import load_config
cfg = load_config()

# configuration
NUM_CLIENTS = getattr(cfg.server, 'num_clients', 5)
BATCH_SIZE = getattr(cfg.dataset, 'batch_size', 32)
NUM_TASKS = getattr(cfg.cl, 'num_experiences', 6)  # Default to 6 tasks for rotated mnist

class RotatedMNISTDataset(Dataset):
    """dataset that applies rotation to mnist images"""
    
    def __init__(self, base_dataset, rotation_angle=0):
        self.base_dataset = base_dataset
        self.rotation_angle = rotation_angle
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # Convert tensor to PIL Image for rotation
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.size(0) == 1:
                # Convert CHW to HW
                image_pil = TF.to_pil_image(image)
            else:
                image_pil = TF.to_pil_image(image)
        else:
            image_pil = image
        
        # Apply rotation
        if self.rotation_angle != 0:
            image_pil = TF.rotate(image_pil, self.rotation_angle, fill=0)
        
        # Convert back to tensor
        image = TF.to_tensor(image_pil)
        
        # Normalize
        image = TF.normalize(image, (0.5,), (0.5,))
        
        return image, label

def generate_rotation_angles(num_tasks, max_angle=180):
    """generate rotation angles for each task"""
    angles = []
    
    # first task: no rotation
    angles.append(0)
    
    # subsequent tasks: evenly spaced rotations
    if num_tasks > 1:
        step = max_angle / (num_tasks - 1)
        for i in range(1, num_tasks):
            angle = i * step
            angles.append(angle)
    
    return angles

def load_datasets(partition_id: int):
    """load rotated mnist datasets for continual learning"""
    
    # base transforms (without normalization since we handle it in RotatedMNISTDataset)
    base_transform = transforms.Compose([
        transforms.ToTensor()
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
    
    # generate rotation angles
    rotation_angles = generate_rotation_angles(NUM_TASKS)
    print(f"rotation angles for {NUM_TASKS} tasks: {rotation_angles}")
    
    # create training experiences
    train_experiences = []
    for i, angle in enumerate(rotation_angles):
        rotated_dataset = RotatedMNISTDataset(client_train, angle)
        aval_dataset = AvalancheDataset(rotated_dataset)
        train_experiences.append(aval_dataset)
        print(f"created train task {i}: rotation {angle}° ({len(rotated_dataset)} samples)")
    
    # create test experiences (use full test set for each task)
    test_experiences = []
    for i, angle in enumerate(rotation_angles):
        rotated_test = RotatedMNISTDataset(test_dataset, angle)
        aval_test = AvalancheDataset(rotated_test)
        test_experiences.append(aval_test)
        print(f"created test task {i}: rotation {angle}° ({len(rotated_test)} samples)")
    
    # create benchmark
    benchmark = benchmark_from_datasets(
        train_datasets=train_experiences,
        test_datasets=test_experiences
    )
    
    print(f"created rotated mnist benchmark with {len(train_experiences)} tasks")
    print(f"client {partition_id}: {len(client_train)} training samples")
    
    return benchmark

def get_dataloaders(partition_id: int):
    """get data loaders for rotated mnist continual learning"""
    benchmark = load_datasets(partition_id)
    
    # handle different benchmark attribute names
    if hasattr(benchmark, 'train_stream'):
        train_stream = benchmark.train_stream
        test_stream = benchmark.test_stream
    elif hasattr(benchmark, 'train_datasets_stream'):
        train_stream = benchmark.train_datasets_stream
        test_stream = benchmark.test_datasets_stream
    else:
        raise ValueError(f"Unknown benchmark type: {type(benchmark)}")
    
    # create dataloaders for each experience
    train_loaders = []
    for exp in train_stream:
        train_loaders.append(
            DataLoader(exp.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        )
    
    test_loaders = []
    for exp in test_stream:
        test_loaders.append(
            DataLoader(exp.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        )
    
    return train_loaders, test_loaders 