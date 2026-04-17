# utils/data_loader.py
import torch
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

# Import the dynamic router
from workloads import load_workload


def partition_niid(dataset, num_clients: int, num_classes: int, alpha: float):
    """Distributes dataset indices to clients using a Dirichlet Distribution to simulate stastical heterogenity (Non-IID)."""
    # Extract labels regardless of the torchvision dataset type
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
    else:
        # Fallback
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    client_indices = {i: [] for i in range(num_clients)}

    # Iterate through each class and distribute its samples across clients
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        # Sample proportions from the Dirichlet Distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to index boundaries
        proportions = np.cumsum(proportions) * len(idx_k)
        splits = proportions.astype(int)[:-1]

        # Split Class indices and assign them to clients
        idx_k_split = np.split(idx_k, splits)
        for i in range(num_clients):
            client_indices[i].extend(idx_k_split[i].tolist())

    # Shuffle each client's assigned indices so batches aren't strictly grouped by class
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return [Subset(dataset, client_indices[i]) for i in range(num_clients)]


def partition_iid(dataset, num_clients: int):
    """Standard IID Partitioning"""
    total_size = len(dataset)
    base_size = total_size // num_clients
    remainder = total_size % num_clients
    lengths = [
        base_size + 1 if i < remainder else base_size for i in range(num_clients)
    ]
    return random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))


def partition_dataset(
    dataset,
    num_clients: int,
    num_classes: int,
    partition_type: str = "iid",
    alpha: float = 0.5,
):
    """Split the global dataset into smaller chunks for each vehicle/client."""

    if partition_type == "iid":
        print(f"[Dataset] Partitioning IID..")
        return partition_iid(dataset, num_clients)

    elif partition_type in ["dirichlet", "non_iid", "niid"]:
        print(f"[Dataset] Partitioning NIID (Dirichlet alpha = {alpha})")
        return partition_niid(dataset, num_clients, num_classes, alpha)
    else:
        raise ValueError(f"Partition type '{partition_type}' is not supported.")


def get_data_loaders(cfg, num_clients: int):
    """
    Main entry point for main.py.
    Handles fetching, partitioning, and wrapping datasets into DataLoaders.
    """
    workload_name = cfg.dataset.workload
    data_dir = cfg.dataset.data_dir
    batch_size = cfg.client.batch_size
    partition_type = cfg.dataset.partition_type
    alpha = cfg.dataset.alpha

    # 1. Ask the Workload Router for the raw datasets
    global_train_dataset, global_test_dataset, metadata = load_workload(
        workload_name, data_dir
    )

    num_classes = metadata["num_classes"]

    # 2. Partition the data for the simulated clients
    client_train_subsets = partition_dataset(
        global_train_dataset, num_clients, num_classes, partition_type
    )
    client_test_subsets = partition_dataset(
        global_test_dataset, num_clients, num_classes, partition_type
    )

    # 3. Wrap the subsets in PyTorch DataLoaders
    train_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
        for subset in client_train_subsets
    ]

    test_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=False)
        for subset in client_test_subsets
    ]

    # 4. Create the central evaluation loader for the server
    global_test_loader = DataLoader(global_test_dataset, batch_size=128, shuffle=False)

    return train_loaders, test_loaders, global_test_loader, metadata
