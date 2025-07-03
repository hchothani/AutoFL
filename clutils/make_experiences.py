from torch.utils.data import Subset
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset

from omegaconf import OmegaConf
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
config_path = project_root / "config" / "config.yaml"
cfg = OmegaConf.load(config_path)

NUM_EXP = cfg.cl.num_experiences


def split_dataset(dataset, n_experiences):
    n = len(dataset)
    indices = list(range(n))
    chunk_size = n // n_experiences
    splits = [indices[i*chunk_size:(i+1)*chunk_size] for i in range(n_experiences-1)]
    splits.append(indices[(n_experiences-1)*chunk_size:])  # Last split takes the remainder

    avalanche_experiences = []
    for i, idxs in enumerate(splits):
        subset = Subset(dataset, idxs)
        # Just create AvalancheDataset without task_labels
        av_dataset = AvalancheDataset(subset)
        avalanche_experiences.append(av_dataset)
    
    return avalanche_experiences
