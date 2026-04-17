"""Structured configuration schema for Hydra-driven AutoFL runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from omegaconf import DictConfig, OmegaConf


@dataclass
class RuntimeConfig:
    mode: str = "sync"


@dataclass
class ClientConfig:
    num_cpus: float = 2.0
    num_gpus: float = 0.2
    local_epochs: int = 3
    learning_rate: float = 0.01
    batch_size: int = 32
    simulate_delay: bool = False
    min_delay: float = 0.5
    max_delay: float = 3.0
    falloff: float = 0.0
    max_concurrency: int | None = None


@dataclass
class FedProxConfig:
    mu: float = 0.01


@dataclass
class ScaffoldConfig:
    eta_l: float = 1.0
    eta_g: float = 1.0


@dataclass
class FedNovaConfig:
    momentum: float = 0.9


@dataclass
class FedOptConfig:
    server_optimizer: str = "adam"
    server_lr: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass
class ServerConfig:
    num_clients: int = 5
    num_rounds: int = 10
    fraction_fit: float = 1.0
    fraction_eval: float = 1.0
    min_fit: int = 5
    min_eval: int = 5
    strategy: str = "fedavg"
    fedprox: FedProxConfig = field(default_factory=FedProxConfig)
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)
    fednova: FedNovaConfig = field(default_factory=FedNovaConfig)
    fedopt: FedOptConfig = field(default_factory=FedOptConfig)
    weighted_averaging: bool = True
    differential_privacy: bool = False
    byzantine_robust: bool = False
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    evaluate_every: int = 1
    early_stopping: bool = False
    patience: int = 3


@dataclass
class ModelConfig:
    name: str = "simple_cnn"
    version: str = "v2"
    pretrained: bool = False
    freeze_backbone: bool = False


@dataclass
class DatasetConfig:
    workload: str = "cifar10"
    partition_type: str = "iid"
    data_dir: str = "./data"
    num_classes: int = 10
    in_channels: int = 3
    input_size: int = 32
    alpha: float = 0.5


@dataclass
class AsyncConfig:
    total_train_time: int = 120
    waiting_interval: int = 10
    max_workers: int = 4
    aggregation_strategy: str = "fedasync"
    staleness_alpha: float = 0.5
    fedasync_mixing_alpha: float = 0.9
    fedasync_a: float = 0.5
    use_staleness: bool = True
    use_sample_weighing: bool = True
    send_gradients: bool = False
    server_artificial_delay: bool = False
    is_streaming: bool = False
    client_local_delay: bool = False
    simulate_delay: bool = True
    min_delay: float = 0.5
    max_delay: float = 3.0
    phase_adaptation: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "num_phases": 4,
            "phase_names": ["morning", "midday", "evening", "night"],
            "phase_weights": [0.25, 0.25, 0.25, 0.25],
            "adapter_param_patterns": ["model.fc"],
            "backbone_learning_rate": None,
            "adapter_learning_rate": None,
        }
    )


@dataclass
class LatencyConfig:
    enabled: bool = True
    csv_path: str = "omnet-data/latency_with_10cars2RSU_30.09.2025.csv"
    sampling_mode: str = "chunk"
    scaling_factor: float = 1.0
    threshold_multiplier: float = 10000.0
    drop_behavior: str = "skip"
    sleep: bool = False
    sleep_log: bool = True
    random_seed: int = 42
    max_clients: int = 10
    throughput_floor_kbps: float = 10.0
    upload_multiplier: float = 1.0
    download_multiplier: float = 1.0
    skip_if_slow_aggregation: bool = False
    skip_if_slow_margin: float = 1.0
    log_round_time_variance: bool = False


@dataclass
class WbConfig:
    project: str = "autofl-testing"
    mode: str = "online"
    run_name_keys: List[str] = field(
        default_factory=lambda: ["dataset.workload", "model.name", "runtime.mode"]
    )


@dataclass
class LoggingConfig:
    output_root: str = "outputs"
    save_client_metrics: bool = True
    save_server_metrics: bool = True
    append_run_name_to_dir: bool = True


def apply_structured_schema(cfg: DictConfig) -> DictConfig:
    """Validate and normalize Hydra-composed config with structured sections."""
    cfg.runtime = OmegaConf.merge(
        OmegaConf.structured(RuntimeConfig), cfg.get("runtime", {})
    )
    cfg.client = OmegaConf.merge(
        OmegaConf.structured(ClientConfig), cfg.get("client", {})
    )
    cfg.server = OmegaConf.merge(
        OmegaConf.structured(ServerConfig), cfg.get("server", {})
    )
    cfg.model = OmegaConf.merge(OmegaConf.structured(ModelConfig), cfg.get("model", {}))
    cfg.dataset = OmegaConf.merge(
        OmegaConf.structured(DatasetConfig), cfg.get("dataset", {})
    )
    cfg["async"] = OmegaConf.merge(
        OmegaConf.structured(AsyncConfig), cfg.get("async", {})
    )
    cfg.latency = OmegaConf.merge(
        OmegaConf.structured(LatencyConfig), cfg.get("latency", {})
    )
    cfg.wb = OmegaConf.merge(OmegaConf.structured(WbConfig), cfg.get("wb", {}))
    cfg.logging = OmegaConf.merge(
        OmegaConf.structured(LoggingConfig), cfg.get("logging", {})
    )
    OmegaConf.set_struct(cfg, True)
    return cfg
