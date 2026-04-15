# models/__init__.py

def get_model_class(model_name: str):
    """
    The Plugin Registry: Lazy-loads and returns the uninstantiated PyTorch model class/function.
    """
    model_name = model_name.lower()
    
    if model_name == "simple_cnn":
        from .simple_cnn import SimpleCNN
        return SimpleCNN
        
    elif model_name == "resnet18":
        from .resnet import resnet18
        return resnet18
        
    elif model_name == "resnet34":
        from .resnet import resnet34
        return resnet34
        
    elif model_name == "wide_resnet":
        from .wide_resnet import WideResNet
        return WideResNet
        
    # --- MobileNet Wrappers ---
    elif model_name == "mobilenet_v2":
        from .mobilenet import MobileNet
        def build_mobilenet_v2(num_classes=10, in_channels=3):
            return MobileNet(num_classes=num_classes, in_channels=in_channels, version="v2")
        return build_mobilenet_v2
        
    elif model_name == "mobilenet_v3_small":
        from .mobilenet import MobileNet
        def build_mobilenet_v3_small(num_classes=10, in_channels=3):
            return MobileNet(num_classes=num_classes, in_channels=in_channels, version="v3_small")
        return build_mobilenet_v3_small
        
    elif model_name == "mobilenet_v3_large":
        from .mobilenet import MobileNet
        def build_mobilenet_v3_large(num_classes=10, in_channels=3):
            return MobileNet(num_classes=num_classes, in_channels=in_channels, version="v3_large")
        return build_mobilenet_v3_large
        
    else:
        supported = [
            "simple_cnn", "resnet18", "resnet34", "wide_resnet", 
            "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"
        ]
        raise ValueError(f"Model '{model_name}' not found. Supported options: {supported}")
