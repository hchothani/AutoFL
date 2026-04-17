"""Configuration utilities for AutoFL"""

import os
from omegaconf import OmegaConf


def load_config(config_path: str | None = None):
    """Load configuration from the provided path or default root config."""
    resolved_path = (
        config_path or os.environ.get("AUTOFL_CONFIG_PATH") or "config/config.yaml"
    )

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"No configuration file found at: {resolved_path}")

    return OmegaConf.load(resolved_path)
