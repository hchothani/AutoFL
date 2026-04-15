import ray
import sys
import os
os.environ["RAY_BACKEND_LOG_LEVEL"] = "fatal"
import atexit
from datetime import datetime
from pathlib import Path
import warnings
import torch
import wandb
from omegaconf import OmegaConf

# Keep your existing imports
from utils.model_factory import validate_config, create_model
from utils.latency_simulator import init_runtime_recorder, flush_runtime_recorder

# Import our reorganized modules
from utils.data_loader import get_data_loaders
from runners.async_runner import get_async_config, run_async_simulation
from runners.sync_runner import run_sync_simulation

warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_cfg():
    """Load configuration from base and experiment configs."""
    base_cfg = OmegaConf.load("config/config.yaml")
    exp_cfg = None
    if "--config-path" in sys.argv and "--config-name" in sys.argv:
        cfg_path = sys.argv[sys.argv.index("--config-path") + 1]
        cfg_name = sys.argv[sys.argv.index("--config-name") + 1]
        candidate = os.path.join(cfg_path, f"{cfg_name}.yaml")
        if os.path.isfile(candidate):
            exp_cfg = OmegaConf.load(candidate)
            if "defaults" in exp_cfg:
                del exp_cfg["defaults"]
        else: print(f"[Config] File {candidate} not found. Using base config.")
    
    cfg = OmegaConf.merge(base_cfg, exp_cfg) if exp_cfg else base_cfg
    return cfg

def prepare_run_directory(cfg, is_async: bool) -> Path:
    """Prepare output directory for the run."""
    output_root = Path(cfg.get("logging", {}).get("output_root", "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)

    mode_suffix = "async" if is_async else "sync"

    run_name = cfg.get("wb", {}).get("name", f"autofl_{mode_suffix}").replace(" ", "-").lower()
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    run_dir = output_root / f"{timestamp}_{run_name}_{mode_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def main():
    print("=" * 60)
    print("AutoFL: Federated Learning Simulation")
    print("=" * 60)

    cfg = load_cfg()
    print("Configuration Loaded:\n" + OmegaConf.to_yaml(cfg))
    validate_config(cfg)
    
    is_async = cfg.get("async", {}).get("enabled", False)
    mode_suffix = "async" if is_async else "sync"

    # 1. Setup Environment
    run_dir = prepare_run_directory(cfg, is_async)
    init_runtime_recorder(cfg)
    atexit.register(flush_runtime_recorder)
    
    with open("temp_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Using Ray for homogenous bottlenecks across sync and async
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False
    }

    max_concurrency = cfg.client.get("max_concurrency")
    if max_concurrency is not None:
        total_cpus = max_concurrency * cfg.client.num_cpus
        total_gpus = max_concurrency * cfg.client.num_gpus
        ray_init_args["num_cpus"] = total_cpus
        ray_init_args["num_gpus"] = total_gpus
        print(f"\n[Ray] Manual bottleneck: Capping at {max_concurrency} concurrent vehicles.")
    else:
        print("\n[Ray] Unconstrained mode: Utilizing all available system hardware.")
    
    if not ray.is_initialized():
        ray.init(**ray_init_args)

    # 2. Initialize WandB
    wandb_enabled = cfg.get("wb", {}).get("mode", "online") != "disabled"
    if wandb_enabled:
        wandb.init(
            project=cfg.get("wb", {}).get("project", "autofl-async"),
            name=cfg.get("wb", {}).get("name", "async_run") + f"{mode_suffix}",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.get("wb", {}).get("mode", "online"),
        )
        print(f"[WandB] Initialized: {wandb.run.name}")
    else:
        print("[WandB] Disabled")

    # 3. Hardware & Models
    if cfg.client.num_gpus > 0.0 and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    def model_fn():
        return create_model(cfg)

    # 4. Load Data
    print(f"\nLoading {cfg.dataset.workload} dataset..")
    train_loaders, test_loaders, global_test_loader = get_data_loaders(cfg, cfg.server.num_clients)

    # 5. Route to Runner
    if is_async:
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
            wandb_enabled
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
           wandb_enabled=wandb_enabled
        )


    safe_results={
        "final_loss": float(results["final_loss"]),
        "final_accuracy": float(results["final_accuracy"]),
        "total_updates" : int(results["total_updates"]),
        "elapsed_time": float(results["elapsed_time"])
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
    if os.path.exists("temp_config.yaml"):
        os.remove("temp_config.yaml")

if __name__ == "__main__":
    main()
