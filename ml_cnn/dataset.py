from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import zlib
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Lazy H5 dataset with subject-independent split (no leakage).
# NOTE: In this H5, gaze is stored per-session as a 2D vector (shape (2,)).
# Images are stored per-session as a sequence of frames (shape (T, H, W, C)).

@dataclass(frozen=True)
class H5Sample:
    subject: str
    session: str
    idx: int  # frame index inside image[session]

def _stable_session_seed(base_seed: int, subject: str, session: str) -> int:
    """Stable per-session seed (independent of enumeration order)."""
    key = f"{subject}/{session}".encode("utf-8")
    return (base_seed + zlib.adler32(key)) % (2**32)



def _list_subjects(h5_path: str, subject_prefix: str = "p") -> List[str]:
    with h5py.File(h5_path, "r") as f:
        subjects = sorted([k for k in f.keys() if k.startswith(subject_prefix)])
    return subjects


def _list_sessions(h5_file: h5py.File, subject: str) -> List[str]:
    return list(h5_file[subject]["image"].keys())


def _sample_indices(rng: np.random.Generator, T: int, k: int) -> np.ndarray:
    if T <= 0:
        return np.array([], dtype=np.int64)
    if k <= 1:
        return np.array([int(rng.integers(0, T))], dtype=np.int64)
    if T >= k:
        return rng.choice(T, size=k, replace=False).astype(np.int64)
    return rng.choice(T, size=k, replace=True).astype(np.int64)


def build_sample_index(
    h5_path: str,
    subject_list: Sequence[str],
    k_per_session: int = 1,
    seed: int = 42,
    split: str = "train",  # train/val/test
) -> List[H5Sample]:
    # Scan metadata only; do not load images into RAM.
    # IMPORTANT:
    # - train: randomized sampling (reproducible per run via `seed`)
    # - val/test: deterministic sampling PER SESSION to prevent epoch-to-epoch validation jitter
    split = split.lower().strip()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split={split!r}, expected 'train'/'val'/'test'.")

    rng = np.random.default_rng(seed)
    samples: List[H5Sample] = []
    with h5py.File(h5_path, "r") as f:
        for subject in subject_list:
            if subject not in f:
                continue
            for session in _list_sessions(f, subject):
                imgs = f[subject]["image"][session]
                T_img = int(imgs.shape[0])  # frames per session
                if T_img <= 0:
                    continue

                if split == "train":
                    idxs = _sample_indices(rng, T_img, k_per_session)
                else:
                    # Deterministic per-session indices (stable across runs/platforms)
                    sseed = _stable_session_seed(seed, subject, session)
                    srng = np.random.default_rng(sseed)
                    idxs = _sample_indices(srng, T_img, k_per_session)

                for idx in idxs:
                    samples.append(H5Sample(subject=subject, session=session, idx=int(idx)))
    return samples



def _ensure_nhwc(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img[..., None]
    if img.ndim == 3:
        return img
    raise ValueError(f"Unexpected image ndim={img.ndim}, shape={img.shape}")


class H5EyeGazeDataset(Dataset):
    # Lazy dataset reading from H5 (each process opens its own handle).

    def __init__(
        self,
        h5_path: str,
        samples: Sequence[H5Sample],
        grayscale: bool = True,
        normalize: bool = True,
        resize_hw: Optional[Tuple[int, int]] = None,  # (H, W)
    ):
        self.h5_path = h5_path
        self.samples = list(samples)
        self.grayscale = grayscale
        self.normalize = normalize
        self.resize_hw = resize_hw

        self._h5: Optional[h5py.File] = None  # opened lazily per process

        # Infer output_dim from one session-level gaze vector.
        self.output_dim = 2
        if len(self.samples) > 0:
            with h5py.File(self.h5_path, "r") as f:
                s0 = self.samples[0]
                g = np.asarray(f[s0.subject]["gaze"][s0.session][:])
                g = g.reshape(-1)
                self.output_dim = int(g.shape[0])

    def __len__(self) -> int:
        return len(self.samples)

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def __getitem__(self, i: int):
        s = self.samples[i]
        f = self._get_h5()

        T_img = int(f[s.subject]["image"][s.session].shape[0])
        idx = int(s.idx) % max(1, T_img)

        img = f[s.subject]["image"][s.session][idx]
        gaze = f[s.subject]["gaze"][s.session][:]  # session-level (2,)

        img = np.asarray(img)
        gaze = np.asarray(gaze, dtype=np.float32).reshape(-1)

        img = _ensure_nhwc(img)
        img = img.astype(np.float32)

        # Optional grayscale
        if self.grayscale and img.shape[-1] != 1:
            if img.shape[-1] >= 3:
                img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
                img = img[..., None]
            else:
                img = img[..., :1]

        # Optional resize
        if self.resize_hw is not None:
            H, W = self.resize_hw
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

        # Normalize to [0,1]
        if self.normalize:
            if img.size > 0 and img.max() > 1.0:
                img = img / 255.0

        x = torch.from_numpy(img).permute(2, 0, 1).float()
        y = torch.from_numpy(gaze).float()

        return x, y

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


def load_dataloaders(
    h5_path: str,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_ratio: float = 0.2,
    seed: int = 42,
    k_per_session: int = 1,
    grayscale: bool = True,
    normalize: bool = True,
    resize_hw: Optional[Tuple[int, int]] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    subjects = _list_subjects(h5_path, subject_prefix="p")
    if len(subjects) == 0:
        raise ValueError("No subjects found in H5 (expected keys starting with 'p').")

    train_subjects, test_subjects = train_test_split(
        subjects, test_size=test_size, random_state=seed, shuffle=True
    )

    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=val_ratio, random_state=seed, shuffle=True
    )

    train_samples = build_sample_index(h5_path, train_subjects, k_per_session=k_per_session, seed=seed, split="train")
    val_samples = build_sample_index(h5_path, val_subjects, k_per_session=k_per_session, seed=seed + 1, split="val")
    test_samples = build_sample_index(h5_path, test_subjects, k_per_session=k_per_session, seed=seed + 2, split="test")

    train_dataset = H5EyeGazeDataset(
        h5_path=h5_path, samples=train_samples, grayscale=grayscale, normalize=normalize, resize_hw=resize_hw
    )
    val_dataset = H5EyeGazeDataset(
        h5_path=h5_path, samples=val_samples, grayscale=grayscale, normalize=normalize, resize_hw=resize_hw
    )
    test_dataset = H5EyeGazeDataset(
        h5_path=h5_path, samples=test_samples, grayscale=grayscale, normalize=normalize, resize_hw=resize_hw
    )

    output_dim = train_dataset.output_dim

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, output_dim
