from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    # Runtime assumptions
    fps_assumed: float = 30.0

    # Temporal window (methodology: 2–3 seconds)
    window_seconds: float = 2.5

    # Blink detection (EAR thresholding)
    ear_thresh: float = 0.20
    ear_consec_frames: int = 2

    # Viewing distance calibration (single-point)
    # Z = Z0 * (s0 / s)
    distance_Z0_cm: float = 60.0
    distance_s0_px: float = 120.0

    # Optional for exposing more knobs later (kept for forward-compat)
    landmark_model_path: str = "face_landmarker.task"
    distance_near_cm: float = 40.0
    distance_far_cm: float = 75.0

    # Gaze CNN checkpoint (set to real path to enable gaze)
    gaze_ckpt_path: Optional[str] = "ml_cnn/checkpoints/best_model_custom_k4.pt"