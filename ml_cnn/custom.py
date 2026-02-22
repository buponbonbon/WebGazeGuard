"""
ml_cnn/custom.py

Custom lightweight CNN for gaze regression.

This file was split out from the original model.py to support ablation:
- custom CNN vs. ResNet backbone
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EyeGazeCNN(nn.Module):
    """
    Simple CNN regressor for eye-gaze.

    Args:
        input_channels: 1 for grayscale, 3 for RGB
        output_dim: 2 (gx, gy) or 3 (vector) depending on your label format
    """

    def __init__(self, input_channels: int = 1, output_dim: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.regressor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)
