import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split


class EyeGazeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataloaders(h5_path, batch_size=32, test_size=0.2, val_ratio=0.2):
    images, gazes = [], []

    with h5py.File(h5_path, 'r') as f:
        subjects = sorted([k for k in f.keys() if k.startswith('p')])
        for subject in subjects:
            for session in f[subject]['image'].keys():
                imgs = f[subject]['image'][session][:]
                gaze_vals = f[subject]['gaze'][session][:]
                for img, gaze in zip(imgs, gaze_vals):
                    images.append(img)
                    gazes.append(gaze)

    X = np.array(images, dtype=np.float32)
    y = np.array(gazes, dtype=np.float32)

    X = X / 255.0 if X.max() > 1 else X
    if len(X[0].shape) == 2:
        X = X[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    train_dataset = EyeGazeDataset(X_train, y_train)
    test_dataset = EyeGazeDataset(X_test, y_test)

    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, y.shape[1]
