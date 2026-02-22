"""
vision/head_distance.py

Viewing distance estimation via *single-point real-world calibration*.

We avoid assuming an average interpupillary distance (e.g., 6.3 cm).
Instead, we calibrate using one known distance Z0 (cm) and the observed
inter-ocular pixel distance s0 (px) at that moment.

Definitions:
- s  : inter-ocular distance in pixels for the current frame
- s0 : inter-ocular distance in pixels during calibration
- Z  : estimated camera-to-face distance in cm for the current frame
- Z0 : known camera-to-face distance in cm during calibration

Model (similar triangles / pinhole camera):
    Z ∝ 1 / s

Single-point calibration gives:
    Z = Z0 * (s0 / s)

Notes:
- Requires only *one* real-world measurement (Z0) and does not need the
  true physical eye distance.
- Accuracy depends on stable head pose and consistent landmark detection.
- Use outer eye corners (MediaPipe landmarks 33 and 263) for s.

Typical workflow:
1) Ask user to sit at a known distance Z0 (e.g., 60 cm) measured by ruler.
2) Capture a few frames and compute median s0.
3) Save (Z0, s0) in config.
4) Runtime: compute s each frame and estimate Z.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# MediaPipe Face Mesh indices: outer eye corners (commonly used)
LEFT_OUTER_EYE = 33
RIGHT_OUTER_EYE = 263


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def interocular_px(landmarks, frame_shape) -> float:

    h, w = frame_shape[:2]

    pL = landmarks[LEFT_OUTER_EYE]
    pR = landmarks[RIGHT_OUTER_EYE]

    x1, y1 = float(pL.x) * w, float(pL.y) * h
    x2, y2 = float(pR.x) * w, float(pR.y) * h

    return _euclidean((x1, y1), (x2, y2))


@dataclass(frozen=True)
class DistanceCalib:

    Z0_cm: float
    s0_px: float


def calibrate_from_frames(
    s_values_px: np.ndarray,
    Z0_cm: float,
) -> DistanceCalib:

    s_values_px = np.asarray(s_values_px, dtype=np.float64)
    s_values_px = s_values_px[np.isfinite(s_values_px)]
    if s_values_px.size == 0:
        raise ValueError("No valid s_values_px for calibration.")

    s0 = float(np.median(s_values_px))
    if s0 <= 0:
        raise ValueError("Invalid s0_px computed for calibration.")

    return DistanceCalib(Z0_cm=float(Z0_cm), s0_px=s0)


def estimate_distance_cm(
    s_px: float,
    calib: DistanceCalib,
) -> Optional[float]:

    if s_px is None:
        return None
    s_px = float(s_px)
    if not np.isfinite(s_px) or s_px <= 0:
        return None

    return calib.Z0_cm * (calib.s0_px / s_px)


def categorize_distance(
    distance_cm: Optional[float],
    near_thresh_cm: float = 40.0,
    far_thresh_cm: float = 75.0,
) -> Optional[str]:

    if distance_cm is None or not np.isfinite(distance_cm):
        return None
    if distance_cm < near_thresh_cm:
        return "too_close"
    if distance_cm > far_thresh_cm:
        return "too_far"
    return "normal"
