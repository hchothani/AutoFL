import pytest
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.data_loader import partition_niid, partition_iid


class MockDataset(Dataset):
    """A fake dataset with 1000 items, perfectly balanced across 10 classes."""

    def __init__(self):
        self.data = torch.randn(1000, 3, 32, 32)
        # 100 zeros, 100 ones, 100 twos, etc.
        self.targets = [i // 100 for i in range(1000)]

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def test_iid_partitioning():
    dataset = MockDataset()
    num_clients = 10

    subsets = partition_iid(dataset, num_clients)

    assert len(subsets) == num_clients
    # IID should split 1000 items perfectly into 10 groups of 100
    assert len(subsets[0]) == 100


def test_dirichlet_niid_partitioning():
    dataset = MockDataset()
    num_clients = 10
    num_classes = 10
    alpha = 0.5  # Moderate skew

    subsets = partition_niid(dataset, num_clients, num_classes, alpha)

    assert len(subsets) == num_clients

    # The total number of items across all subsets MUST still equal 1000
    total_items = sum(len(subset) for subset in subsets)
    assert total_items == 1000

    # Because it's Non-IID, it's highly unlikely all clients have exactly 100 items.
    # We assert that there is variance in the lengths.
    lengths = [len(subset) for subset in subsets]
    assert len(set(lengths)) > 1  # Proves the sizes are heterogeneous
