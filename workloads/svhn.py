import os
from torchvision import datasets, transforms

def get_datasets(data_dir: str):
    """
    Downloads and transforms the SVHN (Street View House Numbers) dataset.
    Returns: (train_dataset, test_dataset, metadata)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    svhn_mean = (0.4377, 0.4438, 0.4728)
    svhn_std = (0.1980, 0.2010, 0.1970)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(svhn_mean, svhn_std),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(svhn_mean, svhn_std),
    ])
    
    print("[Dataset] Loading SVHN (Street View House Numbers)...")
    train_dataset = datasets.SVHN(
        root=data_dir, split='train', download=True, transform=train_transform
    )
    test_dataset = datasets.SVHN(
        root=data_dir, split='test', download=True, transform=test_transform
    )

    metadata = {
        "num_classes": 10,
        "in_channels": 3,
        "input_size": 32
    }
    
    return train_dataset, test_dataset, metadata
