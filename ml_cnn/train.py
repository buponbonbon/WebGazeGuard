import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataset import load_dataloaders
from model import EyeGazeCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(h5_path, epochs=50, batch_size=32, lr=1e-3, patience=10):
    train_loader, val_loader, test_loader, output_dim = load_dataloaders(
        h5_path, batch_size=batch_size
    )

    model = EyeGazeCNN(output_dim=output_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(X), y).item() * X.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{epochs}] Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping triggered')
                break

    torch.save(model.state_dict(), 'final_model.pt')
    print('Final model saved as final_model.pt')

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            preds = model(X.to(DEVICE)).cpu().numpy()
            y_pred.append(preds)
            y_true.append(y.numpy())

    y_true = torch.tensor(y_true).view(-1, output_dim).numpy()
    y_pred = torch.tensor(y_pred).view(-1, output_dim).numpy()

    print('Test MSE:', mean_squared_error(y_true, y_pred))
    print('Test MAE:', mean_absolute_error(y_true, y_pred))


if __name__ == '__main__':
    train('MPIIFaceGaze.h5')
