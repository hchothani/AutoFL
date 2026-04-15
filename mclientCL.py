from __future__ import annotations

import warnings
from typing import List

import torch
import gc  # For garbage collection
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, ConfigsRecord

from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets

from clutils.make_experiences import split_dataset
from clients import FlowerClient, initialize_partition_strategies
from config_utils import load_config
from utils.latency_simulator import LatencySimulator

warnings.filterwarnings("ignore")

cfg = load_config()
latency_simulator = LatencySimulator(cfg)


# Import workload based on configuration
if cfg.dataset.workload == "cifar10":
    from workloads.CIFAR10CL import load_datasets
elif cfg.dataset.workload == "cifar100":
    if cfg.cl.strategy == "domain":
        from workloads.CIFAR100DomainCL import load_datasets
    else:
        from workloads.CIFAR100CL import load_datasets
elif cfg.dataset.workload == "bdd100k":
    from workloads.BDD100KDomainCL import load_datasets
elif cfg.dataset.workload == "kitti":
    from workloads.KITTIDomainCL import load_datasets
elif cfg.dataset.workload == "bdd100k_v2":
    from workloads.BDD100KDomainCLV2 import load_datasets
elif cfg.dataset.workload == "kitti_v2":
    from workloads.KITTIDomainCLV2 import load_datasets
elif cfg.dataset.workload == "bdd100k_10k":
    from workloads.BDD100K10kDomainCL import load_datasets
elif cfg.dataset.workload == "permuted_mnist":
    from workloads.PermutedMNIST import load_datasets
elif cfg.dataset.workload == "rotated_mnist":
    from workloads.RotatedMNIST import load_datasets
elif cfg.dataset.workload == "mnist":
    from workloads.MNIST import load_datasets
elif cfg.dataset.workload == "split_cifar10":
    from workloads.SplitCIFAR10 import load_datasets
elif cfg.dataset.workload == "split_cifar100":
    from workloads.SplitCIFAR100 import load_datasets
elif cfg.dataset.workload == "core50":
    from workloads.CORe50 import load_datasets
else:
    raise ValueError(f"Unknown workload: {cfg.dataset.workload}")

# Device placement
if cfg.client.num_gpus > 0.0 and torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"Using GPU: {DEVICE}")
else:
    DEVICE = torch.device("cpu")
    print(f"Using CPU: {DEVICE}")

NUM_CLIENTS = cfg.server.num_clients
NUM_EXP = cfg.cl.num_experiences


def get_model():
    from utils.model_factory import create_model

    return create_model(cfg)


partition_strategies = initialize_partition_strategies(
    lambda: get_model().to(DEVICE),
    NUM_CLIENTS,
)


def _stream_lengths(stream) -> List[int]:
    return [len(exp.dataset) for exp in stream]


# Function that launches a Client
def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    net = get_model().to(DEVICE)
    partition_id = context.node_config["partition-id"]

    dataset_result = load_datasets(partition_id=partition_id)

    if isinstance(dataset_result, tuple):
        train_data, test_data = dataset_result
        train_experiences = split_dataset(train_data, NUM_EXP)
        test_experiences = split_dataset(test_data, NUM_EXP)
        trainlen_per_exp = [len(exp) for exp in train_experiences]
        testlen_per_exp = [len(exp) for exp in test_experiences]
        benchmark = benchmark_from_datasets(
            train=train_experiences, test=test_experiences
        )
    elif isinstance(dataset_result, dict):
        benchmark = dataset_result["benchmark"]
        if hasattr(benchmark, "train_stream"):
            trainlen_per_exp = _stream_lengths(benchmark.train_stream)
            testlen_per_exp = _stream_lengths(benchmark.test_stream)
        elif hasattr(benchmark, "train_datasets_stream"):
            trainlen_per_exp = _stream_lengths(benchmark.train_datasets_stream)
            testlen_per_exp = _stream_lengths(benchmark.test_datasets_stream)
        else:
            raise ValueError(f"Unknown benchmark type: {type(benchmark)}")
    else:
        benchmark = dataset_result
        if hasattr(benchmark, "train_stream"):
            trainlen_per_exp = _stream_lengths(benchmark.train_stream)
            testlen_per_exp = _stream_lengths(benchmark.test_stream)
        elif hasattr(benchmark, "train_datasets_stream"):
            trainlen_per_exp = _stream_lengths(benchmark.train_datasets_stream)
            testlen_per_exp = _stream_lengths(benchmark.test_datasets_stream)
        else:
            raise ValueError(f"Unknown benchmark type: {type(benchmark)}")

    print(
        "------------------------------------------------ClientID: ",
        partition_id,
        "----------------------------------------------",
    )

    strategy_bundle = partition_strategies[partition_id]

    return FlowerClient(
        context=context,
        net=net,
        benchmark=benchmark,
        trainlen_per_exp=trainlen_per_exp,
        testlen_per_exp=testlen_per_exp,
        partition_id=partition_id,
        strategy_bundle=strategy_bundle,
        latency_simulator=latency_simulator,
        cfg=cfg,
        experience_count=NUM_EXP,
    ).to_client()
