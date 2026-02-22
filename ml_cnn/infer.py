"""
ml_cnn/infer.py

Inference helpers for gaze regression.

Backward compatible:
- load_model(model_path, input_channels=1, output_dim=3) still works.

New:
- Optional model_name="custom"|"resnet18"
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

# Support both package import and running as a script from this folder
try:
    from .model import build_model
except Exception:
    from model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str, input_channels: int = 1, output_dim: int = 3, model_name: str = "custom", pretrained: bool = False):
    model = build_model(model_name, input_channels=input_channels, output_dim=output_dim, pretrained=pretrained)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image_path: str, input_channels: int = 1):
    img = Image.open(image_path)
    img = img.convert("L" if input_channels == 1 else "RGB")
    img = np.array(img, dtype=np.float32) / 255.0

    if input_channels == 1:
        img = img[np.newaxis, np.newaxis, ...]  # (1,1,H,W)
    else:
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # (1,3,H,W)

    tensor = torch.from_numpy(img).to(DEVICE)

    with torch.no_grad():
        gaze = model(tensor).cpu().numpy()[0]

    return gaze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="final_model.pt")
    ap.add_argument("--image", default="test_eye.png")
    ap.add_argument("--model", default="custom", choices=["custom", "resnet18"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--input_channels", type=int, default=1, choices=[1, 3])
    ap.add_argument("--output_dim", type=int, default=3)
    args = ap.parse_args()

    m = load_model(args.model_path, input_channels=args.input_channels, output_dim=args.output_dim, model_name=args.model, pretrained=args.pretrained)
    gaze = predict_image(m, args.image, input_channels=args.input_channels)
    print("Predicted gaze:", gaze)


if __name__ == "__main__":
    main()
