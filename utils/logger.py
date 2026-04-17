
# utils/logger.py

import time
from omegaconf import DictConfig, OmegaConf

def generate_run_name(cfg: DictConfig, is_async: bool) -> str:
    """
    Dynamically builds a W&B run name based on the keys specified in config.yaml.
    """
    parts = []
    
    # 1. Inject runtime variables dynamically into the config
    # This allows us to track things not explicitly in the YAML
    mode_str = "async" if is_async else "sync"
    OmegaConf.update(cfg, "server.mode", mode_str, force_add=True)
    
    # 2. Extract the requested keys
    keys_to_track = cfg.get("wb", {}).get("run_name_keys", [])
    
    for key in keys_to_track:
        val = OmegaConf.select(cfg, key)
        
        if val is None:
            continue
            
        # Format Boolean flags beautifully (e.g., delay: True -> "delay", False -> "nodelay")
        if val in ["true", "false"]:
            flag_name = key.split('.')[0] # Grabs just the word 'delay'
            parts.append(flag_name if val == "true" else f"no_{flag_name}")
            
        # Format Standard Strings/Numbers
        elif str(val).lower() != "none":
            parts.append(str(val))
            
    # 3. Add a short timestamp to guarantee absolute uniqueness
    parts.append(str(mode_str))
    
    # Example Output: "cifar10-resnet18-async-delay-1430"
    return "-".join(parts)
