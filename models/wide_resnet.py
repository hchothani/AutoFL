import torch
import torch.nn as nn
import torch.nn.functional as F

# ... (Keep your BasicBlock and NetworkBlock exactly as they were) ...


class WideResNet(nn.Module):
    # Standardized __init__ contract
    def __init__(
        self, num_classes=10, in_channels=3, depth=28, widen_factor=10, dropRate=0.3
    ):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = int((depth - 4) / 6)
        block = BasicBlock

        # DYNAMIC CHANNELS: Replaced hardcoded 3
        self.conv1 = nn.Conv2d(
            in_channels, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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

        # CRITICAL FIX: Replaced F.avg_pool2d(out, 8) with adaptive pooling
        # This prevents crashes on non-32x32 datasets!
        out = F.adaptive_avg_pool2d(out, (1, 1))

        out = out.view(-1, self.nChannels)
        return self.fc(out)
