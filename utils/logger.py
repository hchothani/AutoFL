# utils/logger.py

import time
from omegaconf import DictConfig, OmegaConf


def generate_run_name(cfg: DictConfig, is_async: bool) -> str:
    """
    Dynamically builds a W&B run name based on the keys specified in config.yaml.
    """
    parts = []

    mode_str = "async" if is_async else "sync"

    # 1. Extract the requested keys
    keys_to_track = cfg.wb.run_name_keys

    for key in keys_to_track:
        if key == "runtime.mode":
            val = mode_str
        else:
            val = OmegaConf.select(cfg, key)

        if val is None:
            continue

        # Format Boolean flags beautifully (e.g., delay: True -> "delay", False -> "nodelay")
        if isinstance(val, bool):
            flag_name = key.split(".")[-1]  # Grabs just the word 'delay'
            parts.append(flag_name if val else f"no_{flag_name}")

        # Format Standard Strings/Numbers
        elif str(val).lower() != "none":
            parts.append(str(val))

    # 3. Add a short timestamp to guarantee absolute uniqueness

    # Example Output: "cifar10-resnet18-async-delay-1430"
    return "-".join(parts)
