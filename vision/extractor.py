from __future__ import annotations


import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from core.schemas import CVFeatures
from vision.landmarks import FaceLandmarker, FaceLandmarks
from vision.ear import compute_ear_both_eyes
from vision.blink import BlinkDetector
from vision.head_pose import estimate_head_pose_pnp


# MediaPipe Face Mesh indices: outer eye corners (commonly used)
LEFT_OUTER_EYE = 33
RIGHT_OUTER_EYE = 263


@dataclass(frozen=True)
class DistanceCalib:

    Z0_cm: float
    s0_px: float


def _xy_dict_to_array(xy: dict[int, Tuple[float, float]], n_points: int = 468) -> np.ndarray:
    #Convert {idx:(x,y)} dict to (n_points,2) float32 array.
    arr = np.zeros((n_points, 2), dtype=np.float32)
    for i, (x, y) in xy.items():
        if 0 <= i < n_points:
            arr[i, 0] = float(x)
            arr[i, 1] = float(y)
    return arr

def _interocular_px(lms_xy: np.ndarray) -> float:
    # Inter-ocular pixel distance between landmarks 33 and 263.
    if lms_xy is None or lms_xy.shape[0] <= RIGHT_OUTER_EYE:
        return float("nan")
    return float(np.linalg.norm(lms_xy[LEFT_OUTER_EYE] - lms_xy[RIGHT_OUTER_EYE]))


def _estimate_distance_cm(lms_xy: np.ndarray, calib: DistanceCalib) -> Optional[float]:

    s = _interocular_px(lms_xy)
    if not np.isfinite(s) or s <= 0:
        return None
    if calib.Z0_cm <= 0 or calib.s0_px <= 0:
        return None
    return float(calib.Z0_cm * (calib.s0_px / s))


def _categorize_distance(distance_cm: Optional[float], near_cm: float, far_cm: float) -> Optional[str]:
    if distance_cm is None or not np.isfinite(distance_cm):
        return None
    if distance_cm < near_cm:
        return "too_close"
    if distance_cm > far_cm:
        return "too_far"
    return "normal"


class VisionExtractor:
    def __init__(
        self,
        *,
        landmark_model_path: str = "face_landmarker.task",
        ear_thresh: float = 0.20,
        ear_consec_frames: int = 2,
        focal_scale: float = 1.0,
        distance_calib: Optional[DistanceCalib] = None,
        distance_near_cm: float = 40.0,
        distance_far_cm: float = 75.0,
    ):
        self.landmarker = FaceLandmarker(model_path=landmark_model_path)
        self.blink = BlinkDetector(ear_thresh=ear_thresh, consec_frames=ear_consec_frames)
        self.focal_scale = float(focal_scale)

        self.distance_calib = distance_calib
        self.distance_near_cm = float(distance_near_cm)
        self.distance_far_cm = float(distance_far_cm)

        # pose history for angular velocity gating
        self._prev_pose_ts_ms: Optional[int] = None
        self._prev_yaw: Optional[float] = None
        self._prev_pitch: Optional[float] = None
        self._freeze_blink_until_ms: Optional[int] = None

    def extract(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> CVFeatures:
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        lm: Optional[FaceLandmarks] = self.landmarker.detect_xy(frame_bgr)
        if lm is None:
            return CVFeatures(
                timestamp_ms=timestamp_ms,
                face_detected=False,
                blink=False,
                total_blinks=self.blink.total_blinks,
            )

        lms_xy = _xy_dict_to_array(lm.xy)

        # EAR
        ear_l, ear_r, ear_m = compute_ear_both_eyes(lms_xy)

        # Head pose
        pose = estimate_head_pose_pnp(lms_xy, frame_bgr.shape, focal_scale=self.focal_scale)
        yaw = pose["yaw"] if pose else None
        pitch = pose["pitch"] if pose else None
        roll = pose["roll"] if pose else None

        blink_flag = False

        # check freeze
        freeze_active = (
            self._freeze_blink_until_ms is not None
            and timestamp_ms < self._freeze_blink_until_ms
        )

        if not freeze_active and yaw is not None and pitch is not None:
            if self._prev_pose_ts_ms is not None:
                dt = (timestamp_ms - self._prev_pose_ts_ms) / 1000.0
                if dt > 0 and self._prev_yaw is not None and self._prev_pitch is not None:
                    yaw_vel = abs(yaw - self._prev_yaw) / dt
                    pitch_vel = abs(pitch - self._prev_pitch) / dt

                    # fast head turn -> freeze blink briefly
                    if yaw_vel > 150 or pitch_vel > 150:
                        self._freeze_blink_until_ms = timestamp_ms + 250
                        # prevent detector stuck state
                        self.blink.update(1.0)
                    else:
                        blink_flag = self.blink.update(ear_m)
                else:
                    blink_flag = self.blink.update(ear_m)
            else:
                blink_flag = self.blink.update(ear_m)
        else:
            # during freeze, keep detector stable
            self.blink.update(1.0)

        # update pose history
        self._prev_pose_ts_ms = timestamp_ms
        self._prev_yaw = yaw
        self._prev_pitch = pitch

        # Distance
        distance_cm: Optional[float] = None
        distance_cat: Optional[str] = None
        if self.distance_calib is not None:
            distance_cm = _estimate_distance_cm(lms_xy, self.distance_calib)
            distance_cat = _categorize_distance(
                distance_cm, self.distance_near_cm, self.distance_far_cm
            )

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