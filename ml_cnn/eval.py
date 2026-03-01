import argparse
import numpy as np
import torch

from .dataset import load_dataloaders
from .model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def yaw_pitch_to_unit_vector(yaw_pitch: np.ndarray, *, angles_in_degrees: bool = False) -> np.ndarray:
    """
    Convert (yaw, pitch) to 3D unit gaze direction vectors.

    Assumptions:
    - yaw, pitch are in radians by default (set angles_in_degrees=True if in degrees).
    - yaw: left/right, pitch: up/down.
    """
    a = yaw_pitch.astype(np.float64)
    yaw = a[:, 0]
    pitch = a[:, 1]

    if angles_in_degrees:
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)

    # Spherical coordinates to Cartesian
    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)

    v = np.stack([x, y, z], axis=1)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return (v / n).astype(np.float64)


def angular_error_yaw_pitch(pred: np.ndarray, gt: np.ndarray, *, angles_in_degrees: bool = False) -> np.ndarray:
    """
    Angular error (degrees) between predicted and ground-truth gaze given as (yaw, pitch).
    """
    vp = yaw_pitch_to_unit_vector(pred, angles_in_degrees=angles_in_degrees)
    vg = yaw_pitch_to_unit_vector(gt, angles_in_degrees=angles_in_degrees)

    dot = np.sum(vp * vg, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def evaluate(model, loader, *, angles_in_degrees: bool = False):
    preds = []
    gts = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            preds.append(out.detach().cpu().numpy())
            gts.append(y.numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    mse = float(np.mean((preds - gts) ** 2))
    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(mse))

    mae_per_dim = np.mean(np.abs(preds - gts), axis=0).astype(float)

    ang = angular_error_yaw_pitch(preds, gts, angles_in_degrees=angles_in_degrees)
    ang_mean = float(np.mean(ang))
    ang_median = float(np.median(ang))

    return mse, mae, rmse, mae_per_dim, ang_mean, ang_median


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt).")
    ap.add_argument("--model", default="custom", choices=["custom", "resnet18"], help="Backbone.")
    ap.add_argument("--h5", required=True, help="Path to H5 dataset.")
    ap.add_argument("--resize", default="", help="Resize H,W e.g. 64,64 (optional).")
    ap.add_argument("--angles_in_degrees", action="store_true", help="Use if your yaw/pitch labels are in degrees.")
    ap.add_argument("--input_channels", type=int, default=1, choices=[1, 3], help="Must match training.")
    args = ap.parse_args()

    resize_hw = None
    if args.resize:
        h, w = [int(x) for x in args.resize.split(",")]
        resize_hw = (h, w)

    _, _, test_loader, output_dim = load_dataloaders(
        h5_path=args.h5,
        resize_hw=resize_hw,
    )

    if output_dim != 2:
        raise SystemExit(
            f"Expected output_dim=2 (yaw,pitch) but got {output_dim}. "
            "If your labels are not yaw/pitch, don't use this script."
        )

    model = build_model(args.model, input_channels=args.input_channels, output_dim=output_dim)

    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    model.load_state_dict(ckpt, strict=True)
    model.to(DEVICE)

    mse, mae, rmse, mae_per_dim, ang_mean, ang_median = evaluate(
        model, test_loader, angles_in_degrees=args.angles_in_degrees
    )

    print("========== Evaluation (yaw/pitch) ==========")
    print("Test MSE :", mse)
    print("Test MAE :", mae)
    print("Test RMSE:", rmse)
    print(f"MAE yaw  : {mae_per_dim[0]:.6f}")
    print(f"MAE pitch: {mae_per_dim[1]:.6f}")
    print(f"Angular Error mean (deg)  : {ang_mean:.4f}")
    print(f"Angular Error median (deg): {ang_median:.4f}")


if __name__ == "__main__":
    main()
