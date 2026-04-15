import os
from torchvision import datasets, transforms

def get_datasets(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)), # CRITICAL: Standardizes variable image sizes
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Simulates daytime/nighttime
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)), # Must match train dimensions
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])
    
    print("[Dataset] Loading GTSRB (Traffic Signs)...")
    # Note: GTSRB uses 'split' instead of 'train=True/False' in Torchvision
    train_dataset = datasets.GTSRB(root=data_dir, split='train', download=True, transform=train_transform)
    test_dataset = datasets.GTSRB(root=data_dir, split='test', download=True, transform=test_transform)

    metadata = {
        "num_classes": 43,
        "in_channels": 3,
        "input_size": 32
    }
    
    return train_dataset, test_dataset, metadata
