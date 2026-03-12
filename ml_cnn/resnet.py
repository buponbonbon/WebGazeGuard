

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torchvision import models
except Exception as e:
    models = None  # will raise in builder


class ResNetGaze(nn.Module):
    def __init__(self, backbone: nn.Module, output_dim: int):
        super().__init__()
        self.backbone = backbone
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_resnet18_gaze(
    *,
    input_channels: int = 1,
    output_dim: int = 3,
    pretrained: bool = False,
) -> ResNetGaze:

    if models is None:
        raise ImportError("torchvision is required for ResNet models but is not available.")

    # Torchvision API differs across versions; handle both.
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
    except Exception:
        net = models.resnet18(pretrained=pretrained)

    # Adapt first conv to grayscale if needed
    if input_channels == 1:
        old_conv = net.conv1
        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        # Initialize weights: average RGB channels if available
        with torch.no_grad():
            if old_conv.weight.shape[1] == 3:
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            else:
                # fallback (shouldn't happen for ResNet18)
                new_conv.weight[:] = old_conv.weight[:, :1, :, :]
        net.conv1 = new_conv
    elif input_channels != 3:
        raise ValueError("input_channels must be 1 or 3 for ResNet.")

    # Replace classification head with regression head
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, int(output_dim))

    return ResNetGaze(net, output_dim=int(output_dim))
