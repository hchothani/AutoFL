import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from omegaconf import DictConfig

def get_data_loaders(cfg: DictConfig, num_clients: int):
    """Create train/test data loaders for each client."""
    
    # Get transforms based on dataset
    if cfg.dataset.workload in ["cifar10", "cifar100"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if cfg.dataset.workload == "cifar10":
            train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        else:
            train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
            
    elif cfg.dataset.workload in ["mnist", "permuted_mnist"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.workload}")

    # Split training data among clients (IID split)
    samples_per_client = len(train_dataset) // num_clients
    client_sizes = [samples_per_client] * num_clients
    # Add remaining samples to last client
    client_sizes[-1] += len(train_dataset) - sum(client_sizes)

    client_datasets = random_split(train_dataset, client_sizes)

    batch_size = cfg.client.batch_size
    train_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        for ds in client_datasets
    ]

    # Each client gets full test set for simplicity
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loaders = [test_loader] * num_clients

    return train_loaders, test_loaders, test_loader
