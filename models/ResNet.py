import torch
import torch.nn as nn
from torchvision import models
from omegaconf import OmegaConf
from pathlib import Path
from models.ResnetModel import resnet50 as resnet50_custom, resnet34 as resnet34_custom, resnet18 as resnet18_custom

# Setup Config
config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
cfg = OmegaConf.load(config_path)

class ResNet(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet, self).__init__()
        if num_classes is None:
            num_classes = cfg.model.num_classes
        # Select architecture; default to resnet18, can be overridden via cfg.model.arch (e.g., 'resnet50')
        # arch = getattr(cfg.model, "arch", "resnet18")
        # self.model = getattr(models, arch)(pretrained=False)
        # self.model = models.resnet18(pretrained=False)    
    
        # CIFAR stem modifications: 3x3 conv, stride=1, padding=1; remove initial maxpool
        # self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.model.maxpool = nn.Identity()

        # Classifier head
        # in_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(in_features, num_classes)

        # using the custom resnet50 model defined in ResnetModel.py
        self.model = resnet18_custom()

    def forward(self, x):
        return self.model(x)

def ResNet18(num_classes=10):
    return ResNet(num_classes=num_classes)

def ResNet34(num_classes=10):
    return ResNet(num_classes=num_classes) 