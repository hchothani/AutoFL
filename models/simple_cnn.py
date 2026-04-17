import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """A mathematically pure CNN that adapts to any input size and channels."""

    def __init__(
        self, num_classes: int = 10, in_channels: int = 3, input_size: int = 32
    ):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Automatically calculate the flattening dimension
        self.feature_size = self._calculate_conv_output_size(in_channels, input_size)

        self.fc1 = nn.Linear(self.feature_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def _calculate_conv_output_size(self, in_channels, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_size, input_size)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.feature_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
