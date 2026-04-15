import os
from torchvision import datasets, transforms

def get_datasets(data_dir: str):
    """
    Downloads and transforms the CIFAR-10 dataset.
    Returns: (train_dataset, test_dataset)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Standard CIFAR-10 data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Pure normalization for evaluation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    print("[Dataset] Loading CIFAR-10...")
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    return train_dataset, test_dataset
