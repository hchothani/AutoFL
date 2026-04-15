
import sys
import os
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
    
    cfg = OmegaConf.merge(base_cfg, exp_cfg) if exp_cfg else base_cfg
    return cfg

def prepare_run_directory(cfg) -> Path:
    """Prepare output directory for the run."""
    output_root = Path(cfg.get("logging", {}).get("output_root", "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = cfg.get("wb", {}).get("name", "autofl_async").replace(" ", "-").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{timestamp}_{run_name}_async"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def main():
    print("=" * 60)
    print("AutoFL: Federated Learning Simulation")
    print("=" * 60)

    cfg = load_cfg()
    validate_config(cfg)
    
    # 1. Setup Environment
    run_dir = prepare_run_directory(cfg)
    init_runtime_recorder(cfg)
    atexit.register(flush_runtime_recorder)
    
    with open("temp_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # 2. Initialize WandB
    wandb_enabled = cfg.get("wb", {}).get("mode", "online") != "disabled"
    if wandb_enabled:
        wandb.init(
            project=cfg.get("wb", {}).get("project", "autofl-async"),
            name=cfg.get("wb", {}).get("name", "async_run") + "_async",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.get("wb", {}).get("mode", "online"),
        )

    # 3. Hardware & Models
    device = torch.device("cuda:0" if cfg.client.num_gpus > 0.0 and torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    def model_fn():
        return create_model(cfg)

    # 4. Load Data
    train_loaders, test_loaders, global_test_loader = get_data_loaders(cfg, cfg.server.num_clients)

    # 5. Route to Runner
    is_async = cfg.get("async", {}).get("enabled", False)
    if is_async:
        async_cfg = get_async_config(cfg)
        results = run_async_simulation(
            cfg, async_cfg, model_fn, train_loaders, test_loaders, global_test_loader, device, wandb_enabled
        )
        
        # Save results
        results_path = run_dir / "async_results.yaml"
        with open(results_path, "w") as f:
            OmegaConf.save(OmegaConf.create(results), f)
        print(f"\\nResults saved to: {results_path}")

    flush_runtime_recorder()
    if wandb_enabled:
        wandb.finish()
    if os.path.exists("temp_config.yaml"):
        os.remove("temp_config.yaml")

if __name__ == "__main__":
    main()
