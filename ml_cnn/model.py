"""
ml_cnn/model.py

Model factory for gaze regression.

This keeps backward compatibility:
- EyeGazeCNN is still importable from model.py (legacy code expects it)

New:
- build_model(name=...) to switch between custom CNN and ResNet backbones.
"""

from __future__ import annotations

from typing import Literal

import torch.nn as nn

from .custom import EyeGazeCNN
from .resnet import build_resnet18_gaze, ResNetGaze


ModelName = Literal["custom", "resnet18"]


def build_model(
    name: ModelName = "custom",
    *,
    input_channels: int = 1,
    output_dim: int = 3,
    pretrained: bool = False,
) -> nn.Module:
    """
    Build gaze regression model by name.

    Args:
        name: "custom" or "resnet18"
        input_channels: 1 (grayscale) or 3 (RGB)
        output_dim: 2 or 3 depending on your labels
        pretrained: only affects resnet18 backbone

    Returns:
        torch.nn.Module
    """
    name = str(name).lower().strip()
    if name in ("custom", "eyegazecnn"):
        return EyeGazeCNN(input_channels=input_channels, output_dim=output_dim)
    if name in ("resnet", "resnet18"):
        return build_resnet18_gaze(input_channels=input_channels, output_dim=output_dim, pretrained=pretrained)
    raise ValueError(f"Unknown model name: {name}. Use 'custom' or 'resnet18'.")
