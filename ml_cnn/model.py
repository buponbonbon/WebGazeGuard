
from __future__ import annotations

from typing import Literal
import torch.nn as nn

from ml_cnn.custom import EyeGazeCNN
from ml_cnn.resnet import build_resnet18_gaze


ModelName = Literal["custom", "resnet18"]


def build_model(
    name: ModelName = "custom",
    *,
    input_channels: int = 1,
    output_dim: int = 3,
    pretrained: bool = False,
) -> nn.Module:

    name = str(name).lower().strip()
    if name in ("custom", "eyegazecnn"):
        return EyeGazeCNN(input_channels=input_channels, output_dim=output_dim)
    if name in ("resnet", "resnet18"):
        return build_resnet18_gaze(input_channels=input_channels, output_dim=output_dim, pretrained=pretrained)
    raise ValueError(f"Unknown model name: {name}. Use 'custom' or 'resnet18'.")
