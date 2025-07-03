import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.dropout_rate = dropout_rate
        self.equalInOut = (in_planes == planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
        
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
            
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
            
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, planes, block, dropout_rate, stride=1):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, planes, nb_layers, dropout_rate, stride)
        
    def _make_layer(self, block, in_planes, planes, nb_layers, dropout_rate, stride):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or planes, planes, dropout_rate, i == 0 and stride or 1))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, dropout_rate, 1)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, dropout_rate, 2)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, dropout_rate, 2)
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def create_wide_resnet(num_classes=100, depth=28, widen_factor=10, dropout=0.3):
    """
    Create Wide ResNet model
    Popular configurations:
    - WRN-28-10: depth=28, widen_factor=10 (36.5M params, great for CIFAR100)
    - WRN-40-2: depth=40, widen_factor=2 (2.2M params, smaller but still good)
    """
    return WideResNet(depth=depth, widen_factor=widen_factor, dropout_rate=dropout, num_classes=num_classes)

# Predefined configurations for easy use
def WideResNet28_10(num_classes=100):
    """Wide ResNet 28-10 - Excellent for CIFAR100 (36.5M parameters)"""
    return create_wide_resnet(num_classes=num_classes, depth=28, widen_factor=10, dropout=0.3)

def WideResNet40_2(num_classes=100):
    """Wide ResNet 40-2 - Smaller but good for CIFAR100 (2.2M parameters)"""
    return create_wide_resnet(num_classes=num_classes, depth=40, widen_factor=2, dropout=0.3)

def WideResNet16_8(num_classes=100):
    """Wide ResNet 16-8 - Medium size for CIFAR100 (11M parameters)"""
    return create_wide_resnet(num_classes=num_classes, depth=16, widen_factor=8, dropout=0.3) 