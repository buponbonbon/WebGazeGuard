from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.schemas import CVFeatures
from .utils_cv import detect_face_landmarks_xy_bgr
from .ear import compute_ear_both_eyes
from .blink import BlinkDetector
from .head_pose import estimate_head_pose_pnp


@dataclass(frozen=True)
class DistanceCalib:
    """
    Single-point real-world calibration for viewing distance.

    Z0_cm: known real distance during calibration (cm), measured by ruler
    s0_px: observed inter-ocular pixel distance at that time (px)
    """
    Z0_cm: float
    s0_px: float


def _interocular_px_from_lms_xy(lms_xy: np.ndarray) -> float:
    """
    Inter-ocular distance in pixels using outer eye corners:
    MediaPipe Face Mesh indices 33 (left) and 263 (right).

    lms_xy: (468, 2) pixel coordinates
    """
    if lms_xy is None or len(lms_xy) <= 263:
        return float("nan")
    pL = lms_xy[33]
    pR = lms_xy[263]
    return float(np.linalg.norm(pL - pR))


def _estimate_distance_cm_from_calib(lms_xy: np.ndarray, calib: DistanceCalib) -> Optional[float]:
    """
    Z = Z0 * (s0 / s), where s is current inter-ocular px distance.
    """
    s = _interocular_px_from_lms_xy(lms_xy)
    if not np.isfinite(s) or s <= 0:
        return None
    if calib.s0_px <= 0 or calib.Z0_cm <= 0:
        return None
    return float(calib.Z0_cm * (calib.s0_px / s))


def _categorize_distance(distance_cm: Optional[float], near_cm: float = 40.0, far_cm: float = 75.0) -> Optional[str]:
    if distance_cm is None or not np.isfinite(distance_cm):
        return None
    if distance_cm < near_cm:
        return "too_close"
    if distance_cm > far_cm:
        return "too_far"
    return "normal"


class VisionExtractor:
    """
    Frame-level vision feature extractor.

    Distance:
      - If distance_calib is provided, estimate viewing distance using:
            Z = Z0 * (s0 / s)
        where s is inter-ocular pixel distance for the current frame.
      - If distance_calib is None, distance fields are returned as None.
    """

    def __init__(
        self,
        *,
        ear_thresh: float = 0.20,
        ear_consec_frames: int = 2,
        focal_scale: float = 1.0,
        distance_calib: Optional[DistanceCalib] = None,
        distance_near_cm: float = 40.0,
        distance_far_cm: float = 75.0,
    ):
        self.blink = BlinkDetector(ear_thresh=ear_thresh, consec_frames=ear_consec_frames)
        self.focal_scale = float(focal_scale)

        self.distance_calib = distance_calib
        self.distance_near_cm = float(distance_near_cm)
        self.distance_far_cm = float(distance_far_cm)

    def extract(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> CVFeatures:
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        lms_xy = detect_face_landmarks_xy_bgr(frame_bgr)
        if lms_xy is None:
            return CVFeatures(
                timestamp_ms=timestamp_ms,
                face_detected=False,
                blink=False,
                total_blinks=self.blink.total_blinks,
            )

        ear_l, ear_r, ear_m = compute_ear_both_eyes(lms_xy)
        blink_flag = self.blink.update(ear_m)

        pose = estimate_head_pose_pnp(lms_xy, frame_bgr.shape, focal_scale=self.focal_scale)
        yaw = pose["yaw"] if pose else None
        pitch = pose["pitch"] if pose else None
        roll = pose["roll"] if pose else None

        distance_cm: Optional[float] = None
        distance_cat: Optional[str] = None
        if self.distance_calib is not None:
            distance_cm = _estimate_distance_cm_from_calib(lms_xy, self.distance_calib)
            distance_cat = _categorize_distance(distance_cm, self.distance_near_cm, self.distance_far_cm)

        return CVFeatures(
            timestamp_ms=timestamp_ms,
            face_detected=True,
            ear_left=ear_l,
            ear_right=ear_r,
            ear_mean=ear_m,
            blink=blink_flag,
            total_blinks=self.blink.total_blinks,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            distance_cm=distance_cm,
            distance_cat=distance_cat,
        )
