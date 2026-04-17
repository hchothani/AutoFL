# utils/model_factory.py

import inspect
from omegaconf import DictConfig
from models import get_model_class


def validate_config(cfg: DictConfig):
    """Ensure the config has the minimum required parameters."""
    if "model" not in cfg or "name" not in cfg.model:
        raise ValueError("[Config Error] Config must contain 'model.name'")


def create_model(cfg: DictConfig):
    """
    Reads the config, dynamically inspects the requested model's signature,
    and instantiates it with perfect precision.
    """
    model_name = cfg.model.name

    # 1. Extract dimensions from validated config
    num_classes = cfg.dataset.num_classes
    in_channels = cfg.dataset.in_channels
    input_size = cfg.dataset.input_size

    # 2. Get the blueprint function/class from our registry
    ModelConstructor = get_model_class(model_name)

    # 3. Build a dictionary of ALL our available dataset traits
    available_kwargs = {
        "num_classes": num_classes,
        "in_channels": in_channels,
        "input_size": input_size,
    }

    # 4. DYNAMIC INSPECTION:
    # Scan the model's __init__ to see which arguments it actually accepts.
    # If the model doesn't want 'input_size' (like ResNet), we simply don't pass it!
    sig = inspect.signature(ModelConstructor)
    valid_kwargs = {
        key: value for key, value in available_kwargs.items() if key in sig.parameters
    }

    # 5. Build and return the model safely
    try:
        model_instance = ModelConstructor(**valid_kwargs)
        return model_instance
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate model '{model_name}': {e}")


def get_model_fn(cfg: DictConfig):
    """
    Creates a closure that returns a fresh model instance.
    This guarantees Ray Actors and Flower Clients can build identical
    copies of the model in their own isolated memory spaces.
    """

    def model_fn():
        return create_model(cfg)

    return model_fn
