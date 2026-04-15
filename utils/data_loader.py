# utils/data_loader.py
import torch
from torch.utils.data import DataLoader, random_split

# Import the dynamic router
from workloads import load_workload

def partition_dataset(dataset, num_clients: int, partition_type: str = "iid"):
    """Split the global dataset into smaller chunks for each vehicle/client."""
    total_size = len(dataset)
    
    if partition_type == "iid":
        base_size = total_size // num_clients
        remainder = total_size % num_clients
        lengths = [base_size + 1 if i < remainder else base_size for i in range(num_clients)]
        return random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
        
    elif partition_type == "non_iid_dirichlet":
        # Future implementation for data heterogeneity
        raise NotImplementedError("Dirichlet non-IID partitioning coming soon.")
    else:
        raise ValueError(f"Partition type '{partition_type}' is not supported.")


def get_data_loaders(cfg, num_clients: int):
    """
    Main entry point for main.py. 
    Handles fetching, partitioning, and wrapping datasets into DataLoaders.
    """
    workload_name = cfg.dataset.workload
    data_dir = cfg.dataset.get("data_dir", "./data")
    batch_size = cfg.client.batch_size
    partition_type = cfg.dataset.get("split", "iid")

    # 1. Ask the Workload Router for the raw datasets
    global_train_dataset, global_test_dataset, metadata = load_workload(workload_name, data_dir)

    # 2. Partition the data for the simulated clients
    client_train_subsets = partition_dataset(global_train_dataset, num_clients, partition_type)
    client_test_subsets = partition_dataset(global_test_dataset, num_clients, partition_type)

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
