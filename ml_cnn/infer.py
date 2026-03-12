

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Support both package import and running as a script from this folder
try:
    from .model import build_model
except Exception:
    from model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Return HxW float32 grayscale image."""
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.ndim == 3:
        # HxWxC
        if img.shape[-1] >= 3:
            g = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            return g.astype(np.float32)
        # Fallback: take first channel
        return img[..., 0].astype(np.float32)
    raise ValueError(f"Unexpected image shape: {img.shape}")


def _resize_2d(img2d: np.ndarray, resize_hw: Tuple[int, int]) -> np.ndarray:
    """Resize HxW image to resize_hw=(H,W)."""
    H, W = resize_hw
    try:
        import cv2  # type: ignore
        return cv2.resize(img2d, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    except Exception:
        # Nearest-ish fallback without cv2
        ys = (np.linspace(0, img2d.shape[0] - 1, H)).astype(np.int64)
        xs = (np.linspace(0, img2d.shape[1] - 1, W)).astype(np.int64)
        return img2d[ys][:, xs].astype(np.float32)


def load_model(
    model_path: str,
    *,
    input_channels: int = 1,
    output_dim: int = 2,
    model_name: str = "custom",
    pretrained: bool = False,
):

    model = build_model(model_name, input_channels=input_channels, output_dim=output_dim, pretrained=pretrained)

    state = torch.load(model_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def predict_array(
    model,
    img_np: np.ndarray,
    *,
    input_channels: int = 1,
    resize_hw: Optional[Tuple[int, int]] = (64, 64),
) -> np.ndarray:

    img = np.asarray(img_np)

    if input_channels == 1:
        img2d = _to_grayscale(img)
        if resize_hw is not None:
            img2d = _resize_2d(img2d, resize_hw)
        # Normalize to [0,1] if needed
        if img2d.size > 0 and img2d.max() > 1.0:
            img2d = img2d / 255.0
        x = torch.from_numpy(img2d).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
    else:
        # RGB path
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=2)
        if resize_hw is not None:
            img_resized = []
            for c in range(3):
                img_resized.append(_resize_2d(img[..., c].astype(np.float32), resize_hw))
            img = np.stack(img_resized, axis=2)
        img = img.astype(np.float32)
        if img.size > 0 and img.max() > 1.0:
            img = img / 255.0
        x = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float()  # (1,3,H,W)

    with torch.no_grad():
        y = model(x.to(DEVICE)).detach().cpu().numpy()[0]
    return y


def predict_image(
    model,
    image_path: str,
    *,
    input_channels: int = 1,
    resize_hw: Optional[Tuple[int, int]] = (64, 64),
) -> np.ndarray:
    """Convenience wrapper to predict from an image file."""
    img = Image.open(image_path).convert("L" if input_channels == 1 else "RGB")
    arr = np.array(img)
    return predict_array(model, arr, input_channels=input_channels, resize_hw=resize_hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt).")

    ap.add_argument("--image", default="", help="Optional: image path to run inference.")
    ap.add_argument("--model", default="custom", choices=["custom", "resnet18"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--input_channels", type=int, default=1, choices=[1, 3])
    ap.add_argument("--output_dim", type=int, default=2)
    ap.add_argument("--resize", default="64,64", help="Resize H,W e.g. 64,64 or empty to disable.")
    args = ap.parse_args()

    resize_hw = None
    if args.resize:
        h, w = [int(x) for x in args.resize.split(",")]
        resize_hw = (h, w)

    m = load_model(
        args.ckpt,
        input_channels=args.input_channels,
        output_dim=args.output_dim,
        model_name=args.model,
        pretrained=args.pretrained,
    )

    if args.image:
        gaze = predict_image(m, args.image, input_channels=args.input_channels, resize_hw=resize_hw)
        print("Predicted gaze:", gaze)
    else:
        print("Model loaded OK. Provide --image to run a quick inference.")


if __name__ == "__main__":
    main()
