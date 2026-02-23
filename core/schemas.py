"""core/schemas.py

Shared dataclasses used to pass structured data between modules.

This file is intentionally lightweight (no heavy imports like torch/cv2) so that
vision/temporal/web code can import it safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class CVFeatures:
    """Frame-level visual features extracted from a single RGB/BGR frame."""

    timestamp_ms: int

    # Detection status
    face_detected: bool = False

    # Eye aspect ratio (EAR)
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None
    ear_mean: Optional[float] = None

    # Blink detection
    blink: bool = False
    total_blinks: int = 0

    # Head pose (degrees, depending on head_pose implementation)
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    # Viewing distance estimation (cm) + category
    distance_cm: Optional[float] = None
    distance_cat: Optional[str] = None  # "too_close" | "normal" | "too_far" | None


@dataclass(frozen=True)
class WindowFeatures:
    """Window-level aggregated features computed over a rolling time window."""

    window_start_ms: int
    window_end_ms: int
    window_seconds: float

    # EAR statistics in the window
    ear_mean: Optional[float] = None
    ear_std: Optional[float] = None

    # Head pose statistics in the window
    yaw_mean: Optional[float] = None
    yaw_std: Optional[float] = None

    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None

    roll_mean: Optional[float] = None
    roll_std: Optional[float] = None

    # Blink rate (blinks per minute)
    blink_rate_bpm: Optional[float] = None

    # Most frequent distance category in the window
    distance_cat_mode: Optional[str] = None


@dataclass(frozen=True)
class NLPFeatures:
    """Text symptom module output (Module 1) used by fusion/risk module (Module 2)."""

    discomfort_level: int
    severity_label: str
    confidence: float
    probabilities: Dict[str, float]
    original_text: str


@dataclass(frozen=True)
class RiskOutput:
    """Module 2 output (placeholder for fusion/risk classifier)."""

    risk_level: str  # e.g., "Low" | "Medium" | "High"
    posture_flag: Optional[str] = None
    explanation: Optional[str] = None
    recommendation: Optional[str] = None
    disclaimer: Optional[str] = None
