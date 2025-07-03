import torch
import torch.nn as nn
import torchvision.models as models

class MobileNet(nn.Module):
    """mobilenet model for continual federated learning"""
    
    def __init__(self, num_classes=10, pretrained=False, version='v2'):
        super(MobileNet, self).__init__()
        
        self.num_classes = num_classes
        self.version = version
        
        if version == 'v2':
            if pretrained:
                self.backbone = models.mobilenet_v2(pretrained=True)
                # replace final classifier
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(self.backbone.last_channel, num_classes)
                )
            else:
                self.backbone = models.mobilenet_v2(pretrained=False, num_classes=num_classes)
        
        elif version == 'v3_small':
            if pretrained:
                self.backbone = models.mobilenet_v3_small(pretrained=True)
                self.backbone.classifier = nn.Sequential(
                    nn.Linear(576, 1024),
                    nn.Hardswish(),
                    nn.Dropout(0.2),
                    nn.Linear(1024, num_classes)
                )
            else:
                self.backbone = models.mobilenet_v3_small(pretrained=False, num_classes=num_classes)
        
        elif version == 'v3_large':
            if pretrained:
                self.backbone = models.mobilenet_v3_large(pretrained=True)
                self.backbone.classifier = nn.Sequential(
                    nn.Linear(960, 1280),
                    nn.Hardswish(),
                    nn.Dropout(0.2),
                    nn.Linear(1280, num_classes)
                )
            else:
                self.backbone = models.mobilenet_v3_large(pretrained=False, num_classes=num_classes)
        else:
            raise ValueError(f"unsupported mobilenet version: {version}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_params_count(self):
        """get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self):
        """estimate memory usage in mb"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

def create_mobilenet(num_classes=10, pretrained=False, version='v2'):
    """factory function to create mobilenet model"""
    return MobileNet(num_classes=num_classes, pretrained=pretrained, version=version) 