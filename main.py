import ray
import os

# Comment below if you want Ray Warnings and Logs
os.environ["RAY_BACKEND_LOG_LEVEL"] = "fatal"

# Patch to ensure padding can be done to datasets -> Caused by old Torch Vesions
import collections
import collections.abc

collections.Sequence = collections.abc.Sequence

import atexit
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import warnings
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

# Local imports
from utils.config_schema import apply_structured_schema
from utils.model_factory import validate_config, get_model_fn
from utils.latency_simulator import init_runtime_recorder, flush_runtime_recorder
from utils.data_loader import get_data_loaders
from utils.logger import generate_run_name
from runners.async_runner import get_async_config, run_async_simulation
from runners.sync_runner import run_sync_simulation

warnings.filterwarnings("ignore", category=DeprecationWarning)


def resolve_runtime_mode(cfg: DictConfig) -> str:
    """Resolve runtime mode using Hydra-composed runtime group."""
    runtime_mode = str(cfg.runtime.mode).strip().lower()
    if runtime_mode not in {"sync", "async"}:
        raise ValueError(
            f"Invalid runtime.mode '{runtime_mode}'. Expected one of: sync, async"
        )
    return runtime_mode


def prepare_run_directory(cfg, runtime_mode: str) -> Path:
    """Prepare output directory for the run."""
    output_root = Path(cfg.logging.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mode_suffix = runtime_mode

    run_name = generate_run_name(cfg, runtime_mode == "async")
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    run_dir = output_root / f"{timestamp}_{run_name}_{mode_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("=" * 60)
    print("AutoFL: Federated Learning Simulation")
    print("=" * 60)

    cfg = apply_structured_schema(cfg)
    print("Configuration Loaded:\n" + OmegaConf.to_yaml(cfg))
    validate_config(cfg)

    runtime_mode = resolve_runtime_mode(cfg)
    mode_suffix = runtime_mode
    is_async = runtime_mode == "async"

    # 1. Setup Environment
    run_dir = prepare_run_directory(cfg, runtime_mode)
    init_runtime_recorder(cfg)
    atexit.register(flush_runtime_recorder)

    # Using Ray for homogenous bottlenecks across sync and async
    ray_init_args = {"ignore_reinit_error": True, "include_dashboard": False}

    max_concurrency = cfg.client.max_concurrency
    if max_concurrency is not None:
        total_cpus = max_concurrency * cfg.client.num_cpus
        total_gpus = max_concurrency * cfg.client.num_gpus
        ray_init_args["num_cpus"] = total_cpus
        ray_init_args["num_gpus"] = total_gpus
        print(
            f"\n[Ray] Manual bottleneck: Capping at {max_concurrency} concurrent vehicles."
        )
    else:
        print("\n[Ray] Unconstrained mode: Utilizing all available system hardware.")

    if not ray.is_initialized():
        ray.init(**ray_init_args)

    # 2. Initialize WandB
    wandb_enabled = cfg.wb.mode != "disabled"
    if wandb_enabled:
        run_name = generate_run_name(cfg, is_async)
        raw_wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(raw_wandb_cfg, dict):
            wandb_cfg: Dict[str, Any] = {str(k): v for k, v in raw_wandb_cfg.items()}
        else:
            wandb_cfg = {}
        wandb.init(
            entity=cfg.wb.entity,
            project=cfg.wb.project,
            name=run_name,
            config=wandb_cfg,
            mode=cfg.wb.mode,
        )
        if wandb.run is not None:
            print(f"[WandB] Initialized: {wandb.run.name}")
        else:
            print("[WandB] Initialized")
    else:
        print("[WandB] Disabled")

    # 3. Hardware & Models
    if cfg.client.num_gpus > 0.0 and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 4. Load Data
    print(f"\n[System] Loading {cfg.dataset.workload} Dataset..")
    train_loaders, test_loaders, global_test_loader, metadata = get_data_loaders(
        cfg, cfg.server.num_clients
    )

    cfg.dataset.num_classes = metadata["num_classes"]
    cfg.dataset.in_channels = metadata["in_channels"]
    cfg.dataset.input_size = metadata["input_size"]

    print(
        f"[System] Auto-configured model inputs: {cfg.dataset.in_channels} channels, {cfg.dataset.num_classes} classes."
    )

    # Loading Models
    model_fn = get_model_fn(cfg)
    test_model = model_fn()
    print(
        f"[Model] Successfully initialized {cfg.model.name} for {cfg.dataset.num_classes} classes."
    )
    del test_model

    # 5. Route to Runner

    if runtime_mode == "async":
        print("\n[Router] Laynching Asynchronous Simulation")
        async_cfg = get_async_config(cfg)
        results = run_async_simulation(
            cfg,
            async_cfg,
            model_fn,
            train_loaders,
            test_loaders,
            global_test_loader,
            device,
            wandb_enabled,
        )

    else:
        print("\n[Router] Launching Synchronous Simulation...")
        results = run_sync_simulation(
            cfg=cfg,
            model_fn=model_fn,
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            global_test_loader=global_test_loader,
            device=device,
            wandb_enabled=wandb_enabled,
        )

    safe_results = {
        "final_loss": float(results["final_loss"]),
        "final_accuracy": float(results["final_accuracy"]),
        "total_updates": int(results["total_updates"]),
        "elapsed_time": float(results["elapsed_time"]),
    }

    results_path = run_dir / f"{mode_suffix}_results.yaml"
    with open(results_path, "w") as f:
        OmegaConf.save(OmegaConf.create(safe_results), f)
        print(f"\\nResults saved to: {results_path}")

    flush_runtime_recorder()
    if ray.is_initialized():
        print("\n[Cleanup] Shutting down Ray cluster")
        ray.shutdown()
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
