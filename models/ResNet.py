import torch
import torch.nn as nn
from torchvision import models
from omegaconf import OmegaConf
from pathlib import Path

# Setup Config
config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
cfg = OmegaConf.load(config_path)

class ResNet(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet, self).__init__()
        if num_classes is None:
            num_classes = cfg.model.num_classes
        self.model = models.resnet18(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def ResNet18(num_classes=10):
    return ResNet(num_classes=num_classes)

def ResNet34(num_classes=10):
    return ResNet(num_classes=num_classes) 