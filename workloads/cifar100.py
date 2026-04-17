import os
from torchvision import datasets, transforms


def get_datasets(data_dir: str):
    """
    Downloads and transforms the CIFAR-100 dataset.
    Returns: (train_dataset, test_dataset)
    """
    os.makedirs(data_dir, exist_ok=True)

    # Aggressive augmentation and CIFAR-100 specific normalization
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )

    # Pure normalization for evaluation
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )

    print("[Dataset] Loading CIFAR-100...")
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    metadata = {"num_classes": 100, "in_channels": 3, "input_size": 32}

    return train_dataset, test_dataset, metadata
