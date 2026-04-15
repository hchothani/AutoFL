import atexit
from datetime import datetime
from pathlib import Path
import sys
import os
import warnings

import torch
import flwr
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp

from omegaconf import OmegaConf

from utils.latency_simulator import init_runtime_recorder, flush_runtime_recorder

# Ignore Deprecation Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ==== Configuration Loading (must happen before imports that use config) ====
def load_cfg():
    base_config_path = "config/config.yaml"
    base_cfg = OmegaConf.load(base_config_path)

    exp_cfg = None
    if "--config-path" in sys.argv and "--config-name" in sys.argv:
        p_idx = sys.argv.index("--config-path") + 1
        n_idx = sys.argv.index("--config-name") + 1
        if p_idx < len(sys.argv) and n_idx < len(sys.argv):
            cfg_path = sys.argv[p_idx]
            cfg_name = sys.argv[n_idx]
            candidate = os.path.join(cfg_path, f"{cfg_name}.yaml")
            if os.path.isfile(candidate):
                exp_cfg = OmegaConf.load(candidate)
                # Remove Hydra-style defaults field if present
                if "defaults" in exp_cfg:
                    del exp_cfg["defaults"]
            else:
                print(f"[Config] File {candidate} not found. Using only base config.")

    if exp_cfg is not None:
        cfg = OmegaConf.merge(base_cfg, exp_cfg)  # exp_cfg overrides base_cfg
        print(f"[Config] Loaded experiment config: {cfg_name}")
        print(f"[Config] Model from experiment: {cfg.model.name}")
    else:
        cfg = base_cfg
        print(f"[Config] Using base config only")
    return cfg


cfg = load_cfg()
print("Configuration Loaded:\n" + OmegaConf.to_yaml(cfg))

# validate configuration
from utils.model_factory import validate_config

validate_config(cfg)


def _sanitize_segment(text: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    text = text.lower().replace(" ", "-")
    return "".join(ch if ch in allowed else "-" for ch in text).strip("-")


def prepare_run_directory(cfg):
    if "logging" not in cfg or cfg.logging is None:
        cfg.logging = OmegaConf.create({})
    if "output_root" not in cfg.logging or cfg.logging.output_root is None:
        cfg.logging.output_root = "outputs"
    output_root = Path(cfg.logging.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = None
    if "wb" in cfg and cfg.wb is not None:
        run_name = cfg.wb.get("name")
    if not run_name:
        dataset_name = cfg.dataset.workload if "dataset" in cfg else "dataset"
        model_name = cfg.model.name if "model" in cfg else "model"
        run_name = f"{dataset_name}_{model_name}"

    run_name = _sanitize_segment(run_name) or "autofl"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latency_cfg = (
        cfg.latency
        if "latency" in cfg and cfg.latency is not None
        else OmegaConf.create({})
    )
    latency_enabled = bool(latency_cfg.get("enabled", False))
    latency_suffix = "latency_on" if latency_enabled else "latency_off"

    append_name = cfg.logging.get("append_run_name_to_dir", True)
    folder_segments = [timestamp]
    if append_name:
        folder_segments.append(run_name)
    folder_segments.append(latency_suffix)
    run_folder = "_".join(seg for seg in folder_segments if seg)
    run_dir = output_root / run_folder
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.logging.run_output_dir = str(run_dir)
    return run_dir


run_dir = prepare_run_directory(cfg)
init_runtime_recorder(cfg)
atexit.register(flush_runtime_recorder)

# Save to temp config for other modules
with open("temp_config.yaml", "w") as f:
    OmegaConf.save(cfg, f)

# Import modules that depend on config AFTER saving it
from mclientCL import client_fn
from mclserver import server_fn


def get_model(cfg):
    """Get model based on configuration"""
    # use intelligent model factory
    from utils.model_factory import create_model

    return create_model(cfg)


def main():
    client = ClientApp(client_fn=client_fn)
    server = ServerApp(server_fn=server_fn)
    backend_config = {
        "client_resources": {
            "num_cpus": cfg.client.num_cpus,
            "num_gpus": cfg.client.num_gpus,
        }
    }

    # Run Simulation
    print("Running Simulation")

    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=cfg.server.num_clients,
        backend_config=backend_config,
    )

    flush_runtime_recorder()

    # Clean up temp config
    if os.path.exists("temp_config.yaml"):
        os.remove("temp_config.yaml")


if __name__ == "__main__":
    main()
