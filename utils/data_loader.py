import torch
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

# Import the dynamic router
from workloads import load_workload

def get_labels(dataset):
    """Recursively extracts labels, handling PyTorch Subsets safely."""
    if isinstance(dataset, Subset):
        parent_labels = get_labels(dataset.dataset)
        return parent_labels[dataset.indices]
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        return np.array(dataset.labels)
    else:
        return np.array([dataset[i][1] for i in range(len(dataset))])

def split_dataset_by_phase(dataset, num_phases: int, num_classes: int):
    """Splits a dataset into phase-specific subsets based on disjoint class labels."""
    if num_phases == 1:
        return [dataset], [list(range(num_classes))]
        
    labels = get_labels(dataset)
    classes_per_phase = max(1, num_classes // num_phases)
    
    phase_datasets = []
    phase_classes_list = []
    
    for p in range(num_phases):
        start_class = p * classes_per_phase
        # Ensure the last phase picks up any remaining classes
        end_class = (p + 1) * classes_per_phase if p < num_phases - 1 else num_classes
        allowed_classes = list(range(start_class, end_class))
        
        mask = np.isin(labels, allowed_classes)
        phase_indices = np.where(mask)[0]
        
        phase_datasets.append(Subset(dataset, phase_indices))
        phase_classes_list.append(allowed_classes)
        
    return phase_datasets, phase_classes_list

def partition_niid(dataset, num_clients: int, phase_classes: list, alpha: float):
    """Distributes dataset indices to clients using a Dirichlet Distribution (Non-IID)."""
    labels = get_labels(dataset)
    client_indices = {i : [] for i in range(num_clients)}

    for k in phase_classes:
        idx_k = np.where(labels == k)[0]
        if len(idx_k) == 0: continue
        
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.cumsum(proportions) * len(idx_k)
        splits = proportions.astype(int)[:-1]

        idx_k_split = np.split(idx_k, splits)
        for i in range(num_clients):
            client_indices[i].extend(idx_k_split[i].tolist())

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return [Subset(dataset, client_indices[i]) for i in range(num_clients)]

def partition_iid(dataset, num_clients: int):
    """Standard IID Partitioning"""
    total_size = len(dataset)
    base_size = total_size // num_clients
    remainder = total_size % num_clients
    lengths = [base_size + 1 if i < remainder else base_size for i in range(num_clients)]
    return random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))

def partition_dataset(dataset, num_clients: int, phase_classes: list, partition_type: str = "iid", alpha: float = 0.5):
    """Split the global dataset into smaller chunks for each vehicle/client."""
    if partition_type == "iid":
        print(f"[Dataset] Partitioning IID..")
        return partition_iid(dataset, num_clients)
    elif partition_type in ["dirichlet", "non_iid", "niid"]:
        print(f"[Dataset] Partitioning NIID (Dirichlet alpha = {alpha})")
        return partition_niid(dataset, num_clients, phase_classes, alpha)
    else:
        raise ValueError(f"Partition type '{partition_type}' is not supported.")

def get_data_loaders(cfg, num_clients: int):
    """Main entry point. Handles fetching, temporal splitting, and partitioning."""
    workload_name = cfg.dataset.workload
    data_dir = cfg.dataset.get("data_dir", "./data")
    batch_size = cfg.client.batch_size
    partition_type = cfg.dataset.get("split", "iid")
    alpha = cfg.dataset.get("alpha", 0.5)

    global_train_dataset, global_test_dataset, metadata = load_workload(workload_name, data_dir)
    num_classes = metadata["num_classes"]

    cl_enabled = cfg.get("cl", {}).get("enabled", False)
    num_phases = cfg.get("cl", {}).get("num_phases", 1) if cl_enabled else 1
    
    # 1. Split temporally into phases
    train_phases, phase_classes_list = split_dataset_by_phase(global_train_dataset, num_phases, num_classes)
    test_phases, _ = split_dataset_by_phase(global_test_dataset, num_phases, num_classes)

    client_train_loaders = [[] for _ in range(num_clients)]
    client_test_loaders = [[] for _ in range(num_clients)]
    
    # 2. Process each phase independently
    for phase_idx in range(num_phases):
        phase_classes = phase_classes_list[phase_idx]
        
        c_train_subsets = partition_dataset(train_phases[phase_idx], num_clients, phase_classes, partition_type, alpha)
        c_test_subsets = partition_dataset(test_phases[phase_idx], num_clients, phase_classes, partition_type, alpha)
        
        for cid in range(num_clients):
            drop_last = len(c_train_subsets[cid]) > batch_size
            client_train_loaders[cid].append(
                DataLoader(c_train_subsets[cid], batch_size=batch_size, shuffle=True, drop_last=drop_last)
            )
            client_test_loaders[cid].append(
                DataLoader(c_test_subsets[cid], batch_size=batch_size, shuffle=False)
            )

    global_test_loader = DataLoader(global_test_dataset, batch_size=128, shuffle=False)

    return client_train_loaders, client_test_loaders, global_test_loader, metadata
