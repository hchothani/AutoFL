from __future__ import annotations

from typing import Optional

from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from omegaconf import DictConfig, OmegaConf


def _select(cfg: DictConfig, key: str, default):
    """Safely select a config value falling back to default when missing."""
    value = OmegaConf.select(cfg, key) if isinstance(cfg, DictConfig) else None
    return default if value is None else value


def build_partitioner(
    cfg: DictConfig,
    *,
    num_partitions: int,
    default_partition_by: str,
    partition_by_override: Optional[str] = None,
):
    """Create an IID or Dirichlet partitioner according to dataset config.

    Args:
        cfg: Global OmegaConf configuration object.
        num_partitions: Number of client partitions to create for the dataset.
        default_partition_by: Fallback label/column name used when config omits one.
        partition_by_override: Explicit column name to partition by. When provided
            it supersedes both the default and the config value.

    Returns:
        A DirichletPartitioner when cfg.dataset.split == "niid", otherwise an
        IidPartitioner. Raises ValueError for unsupported split types.
    """

    split_mode = str(_select(cfg, "dataset.split", "iid")).strip().lower()

    if split_mode == "niid":
        partition_by = partition_by_override or _select(
            cfg, "dataset.niid.partition_by", default_partition_by
        )
        alpha = float(_select(cfg, "dataset.niid.alpha", 0.5))
        seed = int(_select(cfg, "dataset.niid.seed", 42))
        min_partition_size = int(_select(cfg, "dataset.niid.min_partition_size", 1))
        self_balancing = bool(_select(cfg, "dataset.niid.self_balancing", False))

        return DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=partition_by,
            alpha=alpha,
            min_partition_size=min_partition_size,
            self_balancing=self_balancing,
            seed=seed,
        )

    if split_mode == "iid":
        return IidPartitioner(num_partitions=num_partitions)

    raise ValueError(
        f"Unsupported dataset split mode '{split_mode}'. Expected 'iid' or 'niid'."
    )
