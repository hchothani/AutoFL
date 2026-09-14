import os
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

def get_datasets(data_dir: str):
    """
    Downloads and transforms the EuroSAT (RGB) dataset.
    Standardizes 64x64 satellite images to 32x32 for AutoFL models.
    Returns: (train_dataset, test_dataset, metadata)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Standard EuroSAT RGB channel mean and standard deviation
    eurosat_mean = (0.3444, 0.3803, 0.4078)
    eurosat_std = (0.2037, 0.1366, 0.1148)
    
    # 32x32 standard resize to match AutoFL model backbones (SimpleCNN, ResNet18, MobileNetV2)
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(eurosat_mean, eurosat_std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(eurosat_mean, eurosat_std)
    ])
    
    print("[Dataset] Loading EuroSAT (Satellite Images)...")
    train_base = datasets.EuroSAT(root=data_dir, download=True, transform=train_transform)
    test_base = datasets.EuroSAT(root=data_dir, download=False, transform=test_transform)
    
    # Deterministic 80/20 train/test split (21,600 train, 5,400 test)
    num_samples = len(train_base)
    train_len = int(0.8 * num_samples)
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    train_indices = indices[:train_len]
    test_indices = indices[train_len:]
    
    train_dataset = Subset(train_base, train_indices)
    test_dataset = Subset(test_base, test_indices)
    
    metadata = {
        "num_classes": 10,
        "in_channels": 3,
        "input_size": 32
    }
    
    return train_dataset, test_dataset, metadata
