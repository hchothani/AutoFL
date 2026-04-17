import pytest
from omegaconf import OmegaConf
import torch.nn as nn

# Import the actual factory
from utils.model_factory import create_model, validate_config

def test_validate_config_fails_safely():
    """Ensure the factory catches bad configs before allocating memory."""
    bad_cfg = OmegaConf.create({"dataset": {"num_classes": 10}})
    with pytest.raises(ValueError, match="Config must contain 'model.name'"):
        validate_config(bad_cfg)

def test_factory_builds_simple_cnn():
    """SimpleCNN requires 'input_size'. The factory should provide it."""
    cfg = OmegaConf.create({
        "model": {"name": "simple_cnn"},
        "dataset": {"num_classes": 10, "in_channels": 3, "input_size": 32}
    })
    
    model = create_model(cfg)
    assert isinstance(model, nn.Module)
    # The calculated feature size for 32x32 should exist
    assert hasattr(model, 'feature_size') 

def test_factory_builds_resnet_safely():
    """
    ResNet does NOT accept 'input_size' in its __init__. 
    The factory's inspect.signature must catch this and not crash.
    """
    cfg = OmegaConf.create({
        "model": {"name": "resnet18"},
        "dataset": {"num_classes": 43, "in_channels": 1, "input_size": 28} # Simulating MNIST/GTSRB dimensions
    })
    
    # If the factory blindly passed input_size=28, this would throw a TypeError.
    model = create_model(cfg)
    
    assert isinstance(model, nn.Module)
    assert model.fc.out_features == 43 # Proves num_classes was passed successfully
    assert model.conv1[0].in_channels == 1 # Proves the 1-channel dynamic fix worked!

def test_invalid_model_rejection():
    """Ensure the router gracefully rejects hallucinations."""
    cfg = OmegaConf.create({"model": {"name": "fake_transformer"}, "dataset": {}})
    with pytest.raises(RuntimeError, match="Failed to instantiate model"):
        create_model(cfg)
