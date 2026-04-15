from . import cifar10
from . import cifar100
from . import gtsrb
from . import mnist

_WORKLOAD_REGISTRY = {
    "cifar10": cifar10.get_datasets,
    "cifar100": cifar100.get_datasets,
    "gtsrb": gtsrb.get_datasets
    "mnist": mnist.get_datasets
}

def load_workload(workload_name: str, data_dir: str):
    """Dynamically routes to the correct dataset implementation"""
    if workload_name not in _WORKLOAD_REGISTRY:
        supported = ", ".join(_WORKLOAD_REGISTRY.keys())
        raise ValueErorr(f"Workload '{workload_name}' not found. Supported: {supported}")
    dataset_fn = _WORKLOAD_REGISTRY[workload_name]
    return dataset_fn(data_dir)
        
