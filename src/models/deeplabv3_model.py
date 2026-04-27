import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3Model(nn.Module):
    def __init__(self, num_classes=3, pretrained_backbone=False):
        super().__init__()

        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
        )

        # Change first conv layer from 3-channel input to 1-channel input
        old_conv = self.model.backbone.conv1
        self.model.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Replace classifier head for 3 classes
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]