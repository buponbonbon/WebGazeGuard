"""
ml_cnn/dataset.py

Scalable H5 dataset + subject-independent split (no leakage).

Key idea:
- Your H5 contains ~45,000 sessions with a large total number of frames (~20M).
- We should NOT load all frames into RAM.
- We sample K frames per session to control dataset size, and we read samples lazily from H5.

Default behavior:
- Sample k_per_session=1 frame per session (≈ 45k samples)
- Subject-level split: train/val/test subjects are disjoint

You can increase k_per_session (e.g., 2–8) for more diversity without exploding size.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class H5Sample:
    subject: str
    session: str
    idx: int


def _list_subjects(h5_path: str, subject_prefix: str = "p") -> List[str]:
    with h5py.File(h5_path, "r") as f:
        subjects = sorted([k for k in f.keys() if k.startswith(subject_prefix)])
    return subjects


def _list_sessions(h5_file: h5py.File, subject: str) -> List[str]:
    # sessions are keys under f[subject]['image']
    return list(h5_file[subject]["image"].keys())


def _sample_indices(rng: np.random.Generator, T: int, k: int) -> np.ndarray:
    if T <= 0:
        return np.array([], dtype=np.int64)
    if k <= 1:
        return np.array([int(rng.integers(0, T))], dtype=np.int64)
    if T >= k:
        return rng.choice(T, size=k, replace=False).astype(np.int64)
    # if session is short, allow replacement
    return rng.choice(T, size=k, replace=True).astype(np.int64)


def build_sample_index(
    h5_path: str,
    subject_list: Sequence[str],
    k_per_session: int = 1,
    seed: int = 42,
) -> List[H5Sample]:
    """
    Build a list of (subject, session, frame_idx) samples.

    This does NOT load images into memory; it only scans metadata and creates an index.
    """
    rng = np.random.default_rng(seed)
    samples: List[H5Sample] = []
    with h5py.File(h5_path, "r") as f:
        for subject in subject_list:
            if subject not in f:
                continue
            for session in _list_sessions(f, subject):
                imgs = f[subject]["image"][session]
                T = int(imgs.shape[0])
                if T <= 0:
                    continue
                idxs = _sample_indices(rng, T, k_per_session)
                for idx in idxs:
                    samples.append(H5Sample(subject=subject, session=session, idx=int(idx)))
    return samples


def _ensure_nhwc(img: np.ndarray) -> np.ndarray:
    """
    Ensure image shape is (H, W, C). Supports (H,W) or (H,W,C).
    """
    if img.ndim == 2:
        return img[..., None]
    if img.ndim == 3:
        return img
    raise ValueError(f"Unexpected image ndim={img.ndim}, shape={img.shape}")


class H5EyeGazeDataset(Dataset):
    """
    Lazy dataset reading from H5.

    Notes on performance:
    - Keep num_workers=0 for simplest behavior.
    - If you increase num_workers, each worker will open its own H5 handle safely.
    """

    def __init__(
        self,
        h5_path: str,
        samples: Sequence[H5Sample],
        grayscale: bool = True,
        normalize: bool = True,
        # Optional: resize to reduce compute/memory; set None to keep original size
        resize_hw: Optional[Tuple[int, int]] = None,  # (H, W)
    ):
        self.h5_path = h5_path
        self.samples = list(samples)
        self.grayscale = grayscale
        self.normalize = normalize
        self.resize_hw = resize_hw

        self._h5: Optional[h5py.File] = None  # opened lazily per process

        # Infer output_dim by reading one sample gaze vector (if available)
        self.output_dim = 1
        if len(self.samples) > 0:
            with h5py.File(self.h5_path, "r") as f:
                s0 = self.samples[0]
                g = f[s0.subject]["gaze"][s0.session][s0.idx]
                g = np.asarray(g)
                self.output_dim = int(g.shape[-1]) if g.ndim > 0 else 1

    def __len__(self) -> int:
        return len(self.samples)

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            # swmr=True can help for concurrent reads; not required but safe
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def __getitem__(self, i: int):
        s = self.samples[i]
        f = self._get_h5()

        img = f[s.subject]["image"][s.session][s.idx]
        gaze = f[s.subject]["gaze"][s.session][s.idx]

        img = np.asarray(img)
        gaze = np.asarray(gaze, dtype=np.float32)

        # Ensure HWC
        img = _ensure_nhwc(img)

        # Convert to float32
        img = img.astype(np.float32)

        # Optional grayscale (if original has 3 channels)
        if self.grayscale and img.shape[-1] != 1:
            # simple luminance conversion
            if img.shape[-1] >= 3:
                img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
                img = img[..., None]
            else:
                img = img[..., :1]

        # Optional resize
        if self.resize_hw is not None:
            # Use OpenCV if available, else fall back to simple nearest-neighbor via numpy
            H, W = self.resize_hw
            try:
                import cv2

                img2d = img[..., 0] if img.shape[-1] == 1 else img
                resized = cv2.resize(img2d, (W, H), interpolation=cv2.INTER_LINEAR)
                if resized.ndim == 2:
                    resized = resized[..., None]
                img = resized.astype(np.float32)
            except Exception:
                # naive fallback (nearest) to avoid hard dependency
                img2d = img[..., 0]
                ys = (np.linspace(0, img2d.shape[0] - 1, H)).astype(np.int64)
                xs = (np.linspace(0, img2d.shape[1] - 1, W)).astype(np.int64)
                img = img2d[ys][:, xs][..., None].astype(np.float32)

        # Normalize to [0,1] if needed
        if self.normalize:
            if img.max() > 1.0:
                img = img / 255.0

        # To torch: (C,H,W)
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
    """
    Create subject-independent train/val/test loaders.

    Args:
        h5_path: path to H5 file.
        test_size: fraction of subjects for test split.
        val_ratio: fraction of remaining train subjects for val split.
        k_per_session: how many frames to sample per session (1–8 typical).
        grayscale: produce 1-channel images (recommended if your model uses input_channels=1).
        resize_hw: resize images to (H,W) to match your model (e.g., (64,64)).
        num_workers: DataLoader workers; keep 0 for simplest H5 reading.

    Returns:
        train_loader, val_loader, test_loader, output_dim
    """
    subjects = _list_subjects(h5_path, subject_prefix="p")
    if len(subjects) == 0:
        raise ValueError("No subjects found in H5 (expected keys starting with 'p').")

    train_subjects, test_subjects = train_test_split(
        subjects, test_size=test_size, random_state=seed, shuffle=True
    )

    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=val_ratio, random_state=seed, shuffle=True
    )

    # Build sample indices (lazy)
    train_samples = build_sample_index(h5_path, train_subjects, k_per_session=k_per_session, seed=seed)
    val_samples = build_sample_index(h5_path, val_subjects, k_per_session=k_per_session, seed=seed + 1)
    test_samples = build_sample_index(h5_path, test_subjects, k_per_session=k_per_session, seed=seed + 2)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, output_dim
