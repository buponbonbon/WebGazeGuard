
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2


# Default landmark indices for MediaPipe Face Mesh

DEFAULT_LEFT_EYE_IDX: List[int] = [33, 160, 158, 133, 153, 144]
DEFAULT_RIGHT_EYE_IDX: List[int] = [362, 385, 387, 263, 373, 380]


def _to_pixel_points(
    landmarks: Sequence[object],
    indices: Sequence[int],
    frame_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Convert a list of normalized landmarks (with .x, .y in [0,1]) to pixel points."""
    h, w = frame_shape[0], frame_shape[1]
    pts = []
    for i in indices:
        lm = landmarks[i]
        # Support both objects with attributes (mediapipe) and dict-like landmarks
        x = lm.x if hasattr(lm, "x") else lm["x"]
        y = lm.y if hasattr(lm, "y") else lm["y"]
        pts.append((int(x * w), int(y * h)))
    return np.asarray(pts, dtype=np.int32)


def _crop_with_padding(
    frame_bgr: np.ndarray,
    pts_xy: np.ndarray,
    output_size: Tuple[int, int],
    padding: float = 0.35,
) -> np.ndarray:
    """Crop a bounding box around pts with padding, clamp to image, resize. Returns BGR crop resized to output_size."""
    x, y, bw, bh = cv2.boundingRect(pts_xy)

    # padding in pixels
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(frame_bgr.shape[1], x + bw + pad_x)
    y1 = min(frame_bgr.shape[0], y + bh + pad_y)

    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        # Return a black image to avoid crashing downstream
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
    return crop


def crop_eye_regions(
    frame_bgr: np.ndarray,
    landmarks: Sequence[object],
    output_size: Tuple[int, int] = (64, 64),
    left_eye_idx: Optional[Sequence[int]] = None,
    right_eye_idx: Optional[Sequence[int]] = None,
    padding: float = 0.35,
    rgb: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract and resize left/right eye regions from frame based on MediaPipe landmarks."""
    if left_eye_idx is None:
        left_eye_idx = DEFAULT_LEFT_EYE_IDX
    if right_eye_idx is None:
        right_eye_idx = DEFAULT_RIGHT_EYE_IDX

    left_pts = _to_pixel_points(landmarks, left_eye_idx, frame_bgr.shape)
    right_pts = _to_pixel_points(landmarks, right_eye_idx, frame_bgr.shape)

    left_crop = _crop_with_padding(frame_bgr, left_pts, output_size, padding=padding)
    right_crop = _crop_with_padding(frame_bgr, right_pts, output_size, padding=padding)

    if rgb:
        left_crop = cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB)
        right_crop = cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB)

    return left_crop, right_crop
