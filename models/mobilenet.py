import torch
import torch.nn as nn
import torchvision.models as models


class MobileNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, pretrained=False, version="v2"):
        super(MobileNet, self).__init__()
        self.version = version

        if version == "v2":
            self.backbone = models.mobilenet_v2(
                weights="DEFAULT" if pretrained else None
            )

            # DYNAMIC CHANNELS: Alter the first layer if not RGB
            if in_channels != 3:
                orig_conv = self.backbone.features[0][0]
                self.backbone.features[0][0] = nn.Conv2d(
                    in_channels,
                    orig_conv.out_channels,
                    kernel_size=orig_conv.kernel_size,
                    stride=orig_conv.stride,
                    padding=orig_conv.padding,
                    bias=False,
                )

            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(self.backbone.last_channel, num_classes)
            )

        elif version == "v3_small":
            self.backbone = models.mobilenet_v3_small(
                weights="DEFAULT" if pretrained else None
            )
            if in_channels != 3:
                orig_conv = self.backbone.features[0][0]
                self.backbone.features[0][0] = nn.Conv2d(
                    in_channels,
                    orig_conv.out_channels,
                    kernel_size=orig_conv.kernel_size,
                    stride=orig_conv.stride,
                    padding=orig_conv.padding,
                    bias=False,
                )
            self.backbone.classifier = nn.Sequential(
                nn.Linear(576, 1024),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(1024, num_classes),
            )
        else:
            raise ValueError(f"unsupported mobilenet version: {version}")

    def forward(self, x):
        return self.backbone(x)
