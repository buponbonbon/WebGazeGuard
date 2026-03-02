from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

# Support both package import and running as a script from this folder
try:
    from .dataset import load_dataloaders
    from .model import build_model
except Exception:
    from dataset import load_dataloaders
    from model import build_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int = 42) -> None:
    """Make train/val reproducible (as much as possible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN (slower but stable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _clone_loader_no_shuffle(loader: DataLoader) -> DataLoader:
    """
    Ensure validation/test loaders are deterministic.
    If dataset internally samples randomly per __getitem__, you still need to fix that in dataset.py,
    but this removes DataLoader-level shuffling.
    """
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=getattr(loader, "num_workers", 0),
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=False,
    )


def train(
    h5_path: str,
    *,
    model_name: str = "custom",
    pretrained: bool = False,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    input_channels: int = 1,
    resize_hw: tuple[int, int] | None = None,
    k_per_session: int = 1,
    seed: int = 42,
    save_dir: str = "ml_cnn/checkpoints",
) -> None:
    _set_seed(seed)

    train_loader, val_loader, test_loader, output_dim = load_dataloaders(
        h5_path,
        batch_size=batch_size,
        resize_hw=resize_hw,
        k_per_session=k_per_session,
        grayscale=(input_channels == 1),
    )

    # Force deterministic val/test ordering (no shuffle)
    val_loader = _clone_loader_no_shuffle(val_loader)
    test_loader = _clone_loader_no_shuffle(test_loader)

    model = build_model(
        model_name,
        input_channels=input_channels,
        output_dim=output_dim,
        pretrained=pretrained,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)
    tag = f"{model_name}_k{k_per_session}_bs{batch_size}_ep{epochs}_ch{input_channels}"
    best_path = os.path.join(save_dir, f"best_model_{tag}.pt")
    final_path = os.path.join(save_dir, f"final_model_{tag}.pt")

    best_val = float("inf")
    wait = 0

    for epoch in range(epochs):
        # -------- train --------
        model.train()
        train_sse = 0.0
        n_train = 0

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X)

            # Match shapes
            if preds.ndim == 2 and preds.shape[1] == 1 and y.ndim == 1:
                y = y.unsqueeze(1)

            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            # Accumulate sum of squared error (criterion returns mean)
            bs = X.size(0)
            train_sse += loss.item() * bs
            n_train += bs

        train_mse = train_sse / max(1, n_train)

        # -------- validate --------
        model.eval()
        val_sse = 0.0
        n_val = 0
        bad_batches = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = model(X)

                if preds.ndim == 2 and preds.shape[1] == 1 and y.ndim == 1:
                    y = y.unsqueeze(1)

                # Guard against NaN/Inf labels or preds (can explode MSE)
                if not torch.isfinite(y).all() or not torch.isfinite(preds).all():
                    bad_batches += 1
                    continue

                loss = criterion(preds, y)
                bs = X.size(0)
                val_sse += loss.item() * bs
                n_val += bs

        val_mse = val_sse / max(1, n_val)

        msg = f"Epoch [{epoch+1}/{epochs}] Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}"
        if bad_batches > 0:
            msg += f" | skipped_bad_val_batches={bad_batches}"
        print(msg)

        # -------- early stop / save --------
        if val_mse < best_val:
            best_val = val_mse
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    torch.save(model.state_dict(), final_path)
    print(f"Final model saved as {final_path}")
    print(f"Best model saved as  {best_path}")

    # -------- test metrics --------
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            preds = model(X).detach().cpu().numpy()
            y_pred_list.append(preds)
            y_true_list.append(y.numpy())

    y_true = np.concatenate(y_true_list, axis=0) if y_true_list else np.zeros((0, output_dim), dtype=np.float32)
    y_pred = np.concatenate(y_pred_list, axis=0) if y_pred_list else np.zeros((0, output_dim), dtype=np.float32)

    if len(y_true) > 0:
        print("Test MSE :", mean_squared_error(y_true, y_pred))
        print("Test MAE :", mean_absolute_error(y_true, y_pred))
    else:
        print("Test set empty; skipping metrics.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="MPIIFaceGaze.h5", help="Path to H5 dataset.")
    ap.add_argument("--model", default="custom", choices=["custom", "resnet18"], help="Model backbone.")
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained weights (ResNet only).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--input_channels", type=int, default=1, choices=[1, 3])
    ap.add_argument("--resize", default="", help="Resize H,W (e.g., 64,64). Leave empty for no resize.")
    ap.add_argument("--k_per_session", type=int, default=1, help="Frames sampled per session (1–8 typical).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--save_dir", default="ml_cnn/checkpoints", help="Where to save checkpoints.")
    args = ap.parse_args()

    resize_hw = None
    if args.resize:
        h, w = [int(x) for x in args.resize.split(",")]
        resize_hw = (h, w)

    train(
        args.h5,
        model_name=args.model,
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        input_channels=args.input_channels,
        resize_hw=resize_hw,
        k_per_session=args.k_per_session,
        seed=args.seed,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
