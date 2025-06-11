import torch
import torchvision

from omegaconf import OmegaConf
from pathlib import Path

from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset
from avalanche.benchmarks.utils.utils import as_avalanche_dataset

from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark, new_instances_benchmark

project_root = Path(__file__).resolve().parent.parent
config_path = project_root / "config" / "config.yaml"
cfg = OmegaConf.load(config_path)

def make_benchmark(train_dataset, test_dataset):
    bm = new_instances_benchmark(
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        num_experiences = cfg.cl.num_experiences
    )

    return bm
 
