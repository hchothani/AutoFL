import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from pathlib import Path

# Setup Config
config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
cfg = OmegaConf.load(config_path)

class Net(nn.Module):
    def __init__(self, num_classes=None, in_channels=3, input_size=32) -> None:
        super(Net, self).__init__()
        # adaptive input channels for different datasets (RGB=3, grayscale=1)
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # calculate the size after convolutions automatically
        self.feature_size = self._calculate_conv_output_size(input_size)
        
        self.fc1 = nn.Linear(self.feature_size, 120)
        self.fc2 = nn.Linear(120, 84)
        # use parameter if provided, otherwise fall back to config
        if num_classes is None:
            num_classes = cfg.model.num_classes
        self.fc3 = nn.Linear(84, num_classes)
    
    def _calculate_conv_output_size(self, input_size):
        """calculate the flattened size after convolutions"""
        # simulate forward pass through conv layers
        with torch.no_grad():
            # use correct number of input channels
            in_channels = self.conv1.in_channels
            dummy_input = torch.zeros(1, in_channels, input_size, input_size)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()  # total number of elements

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.feature_size)  # use calculated feature size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_simple_cnn(num_classes=10, in_channels=3, input_size=32):
    """factory function to create simple cnn model"""
    return Net(num_classes=num_classes, in_channels=in_channels, input_size=input_size)
