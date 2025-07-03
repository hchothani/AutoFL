import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset, as_avalanche_dataset
from avalanche.benchmarks.utils.data import make_avalanche_dataset

import flwr
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

from omegaconf import OmegaConf
from pathlib import Path

# Setup Config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config
cfg = load_config()

BATCH_SIZE = cfg.dataset.batch_size
NUM_CLIENTS = cfg.server.num_clients

class TupleDataset(torch.utils.data.Dataset):
    """Convert HuggingFace dataset format to tuple format for Avalanche"""
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # CIFAR100 from flwr_datasets uses 'fine_label' instead of 'label'
        return sample["img"], sample["fine_label"]

# Cache FederatedDataset
fds = None

def load_datasets(partition_id: int):
    """Load partitioned CIFAR100"""
    # Only Initialize FederatedDatasaet Once
    global fds
    if fds is None:
        if cfg.dataset.split == "niid":
            partitioner = DirichletPartitioner(
                    num_partitions=NUM_CLIENTS,
                    partition_by="label",
                    alpha=cfg.dataset.niid.alpha,
                    seed=cfg.dataset.niid.seed,
                )
        elif cfg.dataset.split == "iid":
            partitioner = IidPartitioner(num_partitions=NUM_CLIENTS)

        fds = FederatedDataset(dataset="cifar100", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR100(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # Convert to tuple format for Avalanche compatibility
    train_tuple = TupleDataset(partition_train_test["train"])
    test_tuple = TupleDataset(partition_train_test["test"])

    # Wrap in Avalanche Dataset
    train_CIFAR = AvalancheDataset(train_tuple)
    test_CIFAR = AvalancheDataset(test_tuple)

    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return train_CIFAR, test_CIFAR 