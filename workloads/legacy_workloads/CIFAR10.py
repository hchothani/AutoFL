import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr
from flwr_datasets import FederatedDataset

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config
from workloads.partitioning import build_partitioner

cfg = load_config()
NUM_CLIENTS = cfg.server.num_clients
BATCH_SIZE = cfg.dataset.batch_size

fds = None


def load_datasets(partition_id: int):
    global fds
    if fds is None:
        partitioner = build_partitioner(
            cfg,
            num_partitions=NUM_CLIENTS,
            default_partition_by="label",
        )
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader
