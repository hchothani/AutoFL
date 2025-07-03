from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    try:
        net.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"ERROR: Parameter size mismatch detected! This indicates model architecture inconsistency.")
            print(f"Model type: {type(net).__name__}")
            
            # Show model details for debugging
            if hasattr(net, 'backbone'):
                print(f"Model backbone: {type(net.backbone).__name__}")
            if hasattr(net, 'version'):
                print(f"Model version: {net.version}")
            if hasattr(net, 'num_classes'):
                print(f"Model classes: {net.num_classes}")
            
            print(f"Full error: {e}")
            print("This likely means different clients are creating models with different configurations.")
            print("Please ensure all clients use identical model configurations.")
            raise e  # Don't continue with mismatched models
        else:
            raise e


def get_parameters(net) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]
