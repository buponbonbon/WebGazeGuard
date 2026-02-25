from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Support both package import and running as a script from this folder
try:
    from .dataset import load_dataloaders
    from .model import build_model
except Exception:
    from dataset import load_dataloaders
    from model import build_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
) -> None:
    train_loader, val_loader, test_loader, output_dim = load_dataloaders(
        h5_path,
        batch_size=batch_size,
        resize_hw=resize_hw,
        k_per_session=k_per_session,
        grayscale=(input_channels == 1),
    )

    model = build_model(
        model_name,
        input_channels=input_channels,
        output_dim=output_dim,
        pretrained=pretrained,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X)
            if preds.ndim == 2 and preds.shape[1] == 1 and y.ndim == 1:
                y = y.unsqueeze(1)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

        train_loss /= max(1, len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = model(X)
                if preds.ndim == 2 and preds.shape[1] == 1 and y.ndim == 1:
                    y = y.unsqueeze(1)
                val_loss += criterion(preds, y).item() * X.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        print(f"Epoch [{epoch+1}/{epochs}] Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    torch.save(model.state_dict(), "final_model.pt")
    print("Final model saved as final_model.pt")

    # Evaluation
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
        print("Test MSE:", mean_squared_error(y_true, y_pred))
        print("Test MAE:", mean_absolute_error(y_true, y_pred))
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
    )


if __name__ == "__main__":
    main()
