"""Configuration utilities for AutoFL"""

import os
from pathlib import Path
from omegaconf import OmegaConf

def load_config():
    """Load configuration from temporary file if available, otherwise default config"""
    temp_config_path = "temp_config.yaml"
    default_config_path = "config/config.yaml"
    
    if os.path.exists(temp_config_path):
        # Load from temporary config (set by Hydra)
        return OmegaConf.load(temp_config_path)
    elif os.path.exists(default_config_path):
        # Fallback to default config
        return OmegaConf.load(default_config_path)
    else:
        raise FileNotFoundError("No configuration file found") 