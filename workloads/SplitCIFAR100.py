# split cifar100 continual learning workload
# classic class-incremental learning where 100 classes are split across tasks
# model accuracy is not great, but it is a good start, need fixes and few improvements with model architecture

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from flwr_datasets import FederatedDataset
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets

from config_utils import load_config
cfg = load_config()

# configuration
NUM_CLIENTS = getattr(cfg.server, 'num_clients', 5)
BATCH_SIZE = getattr(cfg.dataset, 'batch_size', 32)
NUM_TASKS = getattr(cfg.cl, 'num_experiences', 10)  # Default to 10 tasks for 100 classes

class SplitCIFAR100Dataset(Dataset):
    """dataset that filters cifar100 by specific classes"""
    
    def __init__(self, base_dataset, target_classes, class_mapping=None):
        self.base_dataset = base_dataset
        self.target_classes = set(target_classes)
        self.class_mapping = class_mapping or {}
        
        # filter indices for target classes
        self.filtered_indices = []
        for idx in range(len(base_dataset)):
            _, label = base_dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label in self.target_classes:
                self.filtered_indices.append(idx)
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        orig_idx = self.filtered_indices[idx]
        image, label = self.base_dataset[orig_idx]
        
        # remap label if needed
        if isinstance(label, torch.Tensor):
            label = label.item()
        
        if self.class_mapping:
            label = self.class_mapping.get(label, label)
        
        return image, label

def create_class_splits(num_classes=100, num_tasks=10):
    """create class splits for split cifar100"""
    classes_per_task = num_classes // num_tasks
    splits = []
    
    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        if task_id == num_tasks - 1:
            # last task gets remaining classes
            end_class = num_classes
        else:
            end_class = start_class + classes_per_task
        
        task_classes = list(range(start_class, end_class))
        splits.append(task_classes)
    
    return splits

def load_datasets(partition_id: int):
    """load split cifar100 datasets for continual learning"""
    
    # load federated cifar100 dataset
    fds = FederatedDataset(dataset="cifar100", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    
    # split into train/test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # base transforms
    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch
    
    # apply transforms
    train_data = partition_train_test["train"].with_transform(apply_transforms)
    test_split = fds.load_split("test").with_transform(apply_transforms)
    
    # convert to tuple datasets for easier handling
    class TupleDataset(Dataset):
        def __init__(self, hf_dataset):
            self.hf_dataset = hf_dataset
        
        def __len__(self):
            return len(self.hf_dataset)
        
        def __getitem__(self, idx):
            item = self.hf_dataset[idx]
            return item["img"], item["fine_label"]
    
    train_tuple = TupleDataset(train_data)
    test_tuple = TupleDataset(test_split)
    
    # create class splits
    class_splits = create_class_splits(num_classes=100, num_tasks=NUM_TASKS)
    print(f"class splits for {NUM_TASKS} tasks: {[f'classes {s[0]}-{s[-1]}' for s in class_splits]}")
    
    # create training experiences
    train_experiences = []
    for task_id, task_classes in enumerate(class_splits):
        # create class mapping (map original labels to 0-based for each task)
        class_mapping = {orig_class: new_class for new_class, orig_class in enumerate(task_classes)}
        
        task_dataset = SplitCIFAR100Dataset(train_tuple, task_classes, class_mapping)
        aval_dataset = AvalancheDataset(task_dataset)
        train_experiences.append(aval_dataset)
        
        print(f"train task {task_id}: classes {task_classes[0]}-{task_classes[-1]} -> 0-{len(task_classes)-1} ({len(task_dataset)} samples)")
    
    # create test experiences
    test_experiences = []
    for task_id, task_classes in enumerate(class_splits):
        class_mapping = {orig_class: new_class for new_class, orig_class in enumerate(task_classes)}
        
        task_test = SplitCIFAR100Dataset(test_tuple, task_classes, class_mapping)
        aval_test = AvalancheDataset(task_test)
        test_experiences.append(aval_test)
        
        print(f"test task {task_id}: classes {task_classes[0]}-{task_classes[-1]} -> 0-{len(task_classes)-1} ({len(task_test)} samples)")
    
    # create benchmark
    benchmark = benchmark_from_datasets(
        train_datasets=train_experiences,
        test_datasets=test_experiences
    )
    
    print(f"created split cifar100 benchmark with {len(train_experiences)} tasks")
    
    # Handle different benchmark attribute names
    if hasattr(benchmark, 'train_stream'):
        total_samples = sum(len(exp.dataset) for exp in benchmark.train_stream)
    elif hasattr(benchmark, 'train_datasets_stream'):
        total_samples = sum(len(exp.dataset) for exp in benchmark.train_datasets_stream)
    else:
        total_samples = sum(len(exp) for exp in train_experiences)
    
    print(f"client {partition_id}: {total_samples} total training samples")
    
    return benchmark

def get_dataloaders(partition_id: int):
    """get data loaders for split cifar100 continual learning"""
    benchmark = load_datasets(partition_id)
    
    # handle different benchmark attribute names
    if hasattr(benchmark, 'train_stream'):
        train_stream = benchmark.train_stream
        test_stream = benchmark.test_stream
    elif hasattr(benchmark, 'train_datasets_stream'):
        train_stream = benchmark.train_datasets_stream
        test_stream = benchmark.test_datasets_stream
    else:
        raise ValueError(f"Unknown benchmark type: {type(benchmark)}")
    
    # create dataloaders for each experience
    train_loaders = []
    for exp in train_stream:
        train_loaders.append(
            DataLoader(exp.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        )
    
    test_loaders = []
    for exp in test_stream:
        test_loaders.append(
            DataLoader(exp.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        )
    
    return train_loaders, test_loaders 