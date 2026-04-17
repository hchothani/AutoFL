from pathlib import Path

from hydra import compose, initialize_config_dir

from utils.config_schema import apply_structured_schema


def test_hydra_composes_sync_experiment():
    config_dir = Path(__file__).resolve().parents[1] / "config"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=["experiments=sync_cifar10"])

    cfg = apply_structured_schema(cfg)
    assert cfg.runtime.mode == "sync"
    assert cfg.dataset.workload == "cifar10"
    assert cfg.model.name == "simple_cnn"


def test_hydra_composes_async_experiment():
    config_dir = Path(__file__).resolve().parents[1] / "config"

    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
    ):
        cfg = compose(
            config_name="config",
            overrides=["experiments=async_cifar100_resnet18_gpu"],
        )

    cfg = apply_structured_schema(cfg)
    assert cfg.runtime.mode == "async"
    assert cfg.dataset.workload == "cifar100"
    assert cfg.model.name == "resnet18"
