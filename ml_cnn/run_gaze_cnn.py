"""
ml_cnn/run_gaze_cnn.py

Quick demo / sanity-check runner for gaze CNN.

Why this exists:
- Your H5 can be very large (e.g., ~27GB). We should not load it into memory.
- This script samples a small number of frames lazily from the H5 file (or runs on images)
  and runs inference using either the custom CNN or ResNet-18 model.

Modes:
1) H5 sampling (recommended for huge datasets):
   python ml_cnn/run_gaze_cnn.py --h5 /path/to/data.h5 --ckpt best_model.pt --model custom --num_samples 50

2) Single image:
   python ml_cnn/run_gaze_cnn.py --image path/to/eye.png --ckpt best_model.pt --model resnet18

3) Folder of images:
   python ml_cnn/run_gaze_cnn.py --folder path/to/eyes --ckpt best_model.pt --model custom --limit 100

Outputs:
- Prints predictions to stdout
- Optionally saves a CSV with columns: subject, session, frame_idx, y_true*, y_pred*

Notes:
- Assumes H5 structure like:
    f[subject]['image'][session] -> (T, H, W) or (T, H, W, C)
    f[subject]['gaze'][session]  -> (T, D)
  with subject keys like 'p00', 'p01', ...
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(ckpt_path: str, *, model_name: str, input_channels: int, output_dim: int, pretrained: bool = False):
    # Local import so this script can run either as package or standalone
    try:
        from .model import build_model
    except Exception:
        from model import build_model

    model = build_model(model_name, input_channels=input_channels, output_dim=output_dim, pretrained=pretrained)
    state = torch.load(ckpt_path, map_location=DEVICE)
    # Accept either raw state_dict or checkpoint dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def _ensure_hwc(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img[..., None]
    if img.ndim == 3:
        return img
    raise ValueError(f"Unexpected image shape: {img.shape}")


def _preprocess_np(img: np.ndarray, *, input_channels: int = 1, resize_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Convert numpy image to torch tensor (1,C,H,W), float32, normalized to [0,1].
    """
    img = np.asarray(img)
    img = _ensure_hwc(img).astype(np.float32)

    # Convert to grayscale if needed
    if input_channels == 1 and img.shape[-1] != 1:
        if img.shape[-1] >= 3:
            img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            img = img[..., None]
        else:
            img = img[..., :1]

    if resize_hw is not None:
        H, W = resize_hw
        try:
            import cv2
            img2d = img[..., 0] if img.shape[-1] == 1 else img
            resized = cv2.resize(img2d, (W, H), interpolation=cv2.INTER_LINEAR)
            if resized.ndim == 2:
                resized = resized[..., None]
            img = resized.astype(np.float32)
        except Exception:
            # nearest-neighbor fallback
            img2d = img[..., 0]
            ys = (np.linspace(0, img2d.shape[0] - 1, H)).astype(np.int64)
            xs = (np.linspace(0, img2d.shape[1] - 1, W)).astype(np.int64)
            img = img2d[ys][:, xs][..., None].astype(np.float32)

    # Normalize
    if img.max() > 1.0:
        img = img / 255.0

    # To torch (1,C,H,W)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return x


def _predict(model, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        y = model(x.to(DEVICE)).detach().cpu().numpy()[0]
    return y


def _list_subjects(h5_file, prefix: str = "p") -> List[str]:
    return sorted([k for k in h5_file.keys() if k.startswith(prefix)])


def _list_sessions(h5_file, subject: str) -> List[str]:
    return list(h5_file[subject]["image"].keys())


def _sample_h5(h5_path: str, num_samples: int, seed: int = 42) -> List[Tuple[str, str, int]]:
    """
    Return a list of (subject, session, frame_idx) samples, drawn across sessions.
    """
    import h5py

    rng = np.random.default_rng(seed)
    picks: List[Tuple[str, str, int]] = []

    with h5py.File(h5_path, "r") as f:
        subjects = _list_subjects(f, prefix="p")
        if not subjects:
            raise ValueError("No subject keys found (expected prefix 'p').")

        for _ in range(num_samples):
            subj = subjects[int(rng.integers(0, len(subjects)))]
            sessions = _list_sessions(f, subj)
            if not sessions:
                continue
            sess = sessions[int(rng.integers(0, len(sessions)))]
            T = int(f[subj]["image"][sess].shape[0])
            if T <= 0:
                continue
            idx = int(rng.integers(0, T))
            picks.append((subj, sess, idx))

    return picks


def run_on_h5(
    *,
    h5_path: str,
    ckpt: str,
    model_name: str,
    input_channels: int,
    resize_hw: Optional[Tuple[int, int]],
    num_samples: int,
    seed: int,
    out_csv: Optional[str],
    pretrained: bool,
):
    import h5py

    # Determine output_dim by peeking one gaze vector
    with h5py.File(h5_path, "r") as f:
        subjects = _list_subjects(f, prefix="p")
        subj0 = subjects[0]
        sess0 = _list_sessions(f, subj0)[0]
        g0 = np.asarray(f[subj0]["gaze"][sess0][0])
        output_dim = int(g0.shape[-1]) if g0.ndim > 0 else 1

    model = _load_model(ckpt, model_name=model_name, input_channels=input_channels, output_dim=output_dim, pretrained=pretrained)

    picks = _sample_h5(h5_path, num_samples=num_samples, seed=seed)

    rows = []
    with h5py.File(h5_path, "r") as f:
        for subj, sess, idx in picks:
            img = np.asarray(f[subj]["image"][sess][idx])
            y_true = np.asarray(f[subj]["gaze"][sess][idx], dtype=np.float32)

            x = _preprocess_np(img, input_channels=input_channels, resize_hw=resize_hw)
            y_pred = _predict(model, x)

            print(f"[{subj}/{sess} idx={idx}] true={y_true} pred={y_pred}")

            row = {"subject": subj, "session": sess, "frame_idx": idx}
            for j in range(output_dim):
                row[f"y_true_{j}"] = float(y_true[j])
                row[f"y_pred_{j}"] = float(y_pred[j])
            rows.append(row)

    if out_csv and rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as wf:
            w = csv.DictWriter(wf, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print("Saved:", out_csv)


def run_on_images(
    *,
    image_paths: List[str],
    ckpt: str,
    model_name: str,
    input_channels: int,
    output_dim: int,
    resize_hw: Optional[Tuple[int, int]],
    out_csv: Optional[str],
    pretrained: bool,
):
    from PIL import Image

    model = _load_model(ckpt, model_name=model_name, input_channels=input_channels, output_dim=output_dim, pretrained=pretrained)

    rows = []
    for p in image_paths:
        img = Image.open(p).convert("L" if input_channels == 1 else "RGB")
        arr = np.array(img)

        x = _preprocess_np(arr, input_channels=input_channels, resize_hw=resize_hw)
        y_pred = _predict(model, x)

        print(f"[{p}] pred={y_pred}")

        row = {"path": p}
        for j in range(output_dim):
            row[f"y_pred_{j}"] = float(y_pred[j])
        rows.append(row)

    if out_csv and rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as wf:
            w = csv.DictWriter(wf, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print("Saved:", out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt).")
    ap.add_argument("--model", default="custom", choices=["custom", "resnet18"], help="Backbone.")
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet weights (ResNet only).")

    ap.add_argument("--h5", default="", help="Path to H5 dataset (large file OK; sampled lazily).")
    ap.add_argument("--num_samples", type=int, default=50, help="Random frames to sample from H5.")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--image", default="", help="Single image path.")
    ap.add_argument("--folder", default="", help="Folder of images.")
    ap.add_argument("--limit", type=int, default=200, help="Max images from folder.")

    ap.add_argument("--input_channels", type=int, default=1, choices=[1, 3])
    ap.add_argument("--resize", default="", help="Resize H,W e.g. 64,64 (optional).")
    ap.add_argument("--output_dim", type=int, default=3, help="Output dim for image-only mode.")
    ap.add_argument("--out_csv", default="", help="Optional CSV output path.")

    args = ap.parse_args()

    resize_hw = None
    if args.resize:
        h, w = [int(x) for x in args.resize.split(",")]
        resize_hw = (h, w)

    out_csv = args.out_csv or None

    if args.h5:
        run_on_h5(
            h5_path=args.h5,
            ckpt=args.ckpt,
            model_name=args.model,
            input_channels=args.input_channels,
            resize_hw=resize_hw,
            num_samples=args.num_samples,
            seed=args.seed,
            out_csv=out_csv,
            pretrained=args.pretrained,
        )
        return

    image_paths: List[str] = []
    if args.image:
        image_paths = [args.image]
    elif args.folder:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        files = [str(p) for p in sorted(Path(args.folder).glob("*")) if p.suffix.lower() in exts]
        image_paths = files[: args.limit]

    if image_paths:
        run_on_images(
            image_paths=image_paths,
            ckpt=args.ckpt,
            model_name=args.model,
            input_channels=args.input_channels,
            output_dim=args.output_dim,
            resize_hw=resize_hw,
            out_csv=out_csv,
            pretrained=args.pretrained,
        )
        return

    raise SystemExit("Provide one of: --h5, --image, or --folder")


if __name__ == "__main__":
    main()
