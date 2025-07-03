import torch
import flwr
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp

import sys, os
from omegaconf import OmegaConf
import warnings

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
        server_app = server,
        client_app = client,
        num_supernodes = cfg.server.num_clients,
        backend_config = backend_config,
    )
    
    # Clean up temp config
    if os.path.exists("temp_config.yaml"):
        os.remove("temp_config.yaml")

if __name__ == "__main__":
    main()
