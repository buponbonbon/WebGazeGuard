
"""
run_gaze_cnn.py (fixed)

Sanity-check runner for gaze CNN on huge H5 datasets or images.
Patch: robust gaze reading (supports per-session or per-frame labels).
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
    try:
        from .model import build_model
    except Exception:
        from model import build_model

    model = build_model(model_name, input_channels=input_channels, output_dim=output_dim, pretrained=pretrained)
    state = torch.load(ckpt_path, map_location=DEVICE)

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
    img = np.asarray(img)
    img = _ensure_hwc(img).astype(np.float32)

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
            img2d = img[..., 0]
            ys = (np.linspace(0, img2d.shape[0] - 1, H)).astype(np.int64)
            xs = (np.linspace(0, img2d.shape[1] - 1, W)).astype(np.int64)
            img = img2d[ys][:, xs][..., None].astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

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
    import h5py

    rng = np.random.default_rng(seed)
    picks: List[Tuple[str, str, int]] = []

    with h5py.File(h5_path, "r") as f:
        subjects = _list_subjects(f, prefix="p")

        for _ in range(num_samples):
            subj = subjects[int(rng.integers(0, len(subjects)))]
            sessions = _list_sessions(f, subj)
            sess = sessions[int(rng.integers(0, len(sessions)))]

            T = int(f[subj]["image"][sess].shape[0])
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

    # Robust output_dim detection
    with h5py.File(h5_path, "r") as f:
        subjects = _list_subjects(f)
        subj0 = subjects[0]
        sess0 = _list_sessions(f, subj0)[0]

        g = np.asarray(f[subj0]["gaze"][sess0][...])

        if g.ndim == 1:
            output_dim = int(g.shape[0])
        elif g.ndim == 2:
            output_dim = int(g.shape[1])
        else:
            output_dim = 1

    model = _load_model(
        ckpt,
        model_name=model_name,
        input_channels=input_channels,
        output_dim=output_dim,
        pretrained=pretrained,
    )

    picks = _sample_h5(h5_path, num_samples=num_samples, seed=seed)

    rows = []
    with h5py.File(h5_path, "r") as f:
        for subj, sess, idx in picks:
            img = np.asarray(f[subj]["image"][sess][idx])

            g = np.asarray(f[subj]["gaze"][sess][...], dtype=np.float32)
            y_true = g if g.ndim == 1 else g[idx]

            x = _preprocess_np(img, input_channels=input_channels, resize_hw=resize_hw)
            y_pred = _predict(model, x)

            print(f"[{subj}/{sess} idx={idx}] true={y_true} pred={y_pred}")

            row = {"subject": subj, "session": sess, "frame_idx": idx}
            for j in range(len(y_pred)):
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="custom", choices=["custom", "resnet18"])
    ap.add_argument("--pretrained", action="store_true")

    ap.add_argument("--h5", default="")
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--input_channels", type=int, default=1, choices=[1, 3])
    ap.add_argument("--resize", default="")
    ap.add_argument("--out_csv", default="")

    args = ap.parse_args()

    resize_hw = None
    if args.resize:
        h, w = [int(x) for x in args.resize.split(",")]
        resize_hw = (h, w)

    run_on_h5(
        h5_path=args.h5,
        ckpt=args.ckpt,
        model_name=args.model,
        input_channels=args.input_channels,
        resize_hw=resize_hw,
        num_samples=args.num_samples,
        seed=args.seed,
        out_csv=args.out_csv or None,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
