from __future__ import annotations


import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from core.schemas import CVFeatures
from .landmarks import FaceLandmarker, FaceLandmarks
from .ear import compute_ear_both_eyes
from .blink import BlinkDetector
from .head_pose import estimate_head_pose_pnp


# MediaPipe Face Mesh indices: outer eye corners (commonly used)
LEFT_OUTER_EYE = 33
RIGHT_OUTER_EYE = 263


@dataclass(frozen=True)
class DistanceCalib:

    Z0_cm: float
    s0_px: float


def _xy_dict_to_array(xy: dict[int, Tuple[float, float]], n_points: int = 468) -> np.ndarray:
    """Convert {idx:(x,y)} dict -> (n_points,2) float32 array."""
    arr = np.zeros((n_points, 2), dtype=np.float32)
    for i, (x, y) in xy.items():
        if 0 <= i < n_points:
            arr[i, 0] = float(x)
            arr[i, 1] = float(y)
    return arr


def _interocular_px(lms_xy: np.ndarray) -> float:
    """s: inter-ocular pixel distance between landmarks 33 and 263."""
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

    def extract(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> CVFeatures:
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        lm: Optional[FaceLandmarks] = self.landmarker.detect_xy(frame_bgr)
        if lm is None:
            # Safe fallback when detection fails
            return CVFeatures(
                timestamp_ms=timestamp_ms,
                face_detected=False,
                blink=False,
                total_blinks=self.blink.total_blinks,
            )

        lms_xy = _xy_dict_to_array(lm.xy)

        # EAR + blink
        ear_l, ear_r, ear_m = compute_ear_both_eyes(lms_xy)
        blink_flag = self.blink.update(ear_m)

        # Head pose
        pose = estimate_head_pose_pnp(lms_xy, frame_bgr.shape, focal_scale=self.focal_scale)
        yaw = pose["yaw"] if pose else None
        pitch = pose["pitch"] if pose else None
        roll = pose["roll"] if pose else None

        # Distance (optional)
        distance_cm: Optional[float] = None
        distance_cat: Optional[str] = None
        if self.distance_calib is not None:
            distance_cm = _estimate_distance_cm(lms_xy, self.distance_calib)
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

    def calibrate_distance_webcam(
        self,
        *,
        Z0_cm: float,
        duration_s: float = 10.0,
        camera_id: int = 0,
        show_preview: bool = True,
        min_samples: int = 30,
    ) -> DistanceCalib:

        import cv2  # local import to keep extractor import-light

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera_id={camera_id}")

        s_values: List[float] = []
        t_end = time.time() + float(duration_s)

        try:
            while time.time() < t_end:
                ok, frame = cap.read()
                if not ok:
                    continue

                lm = self.landmarker.detect_xy(frame)
                if lm is not None:
                    lms_xy = _xy_dict_to_array(lm.xy)
                    s = _interocular_px(lms_xy)
                    if np.isfinite(s) and s > 0:
                        s_values.append(float(s))

                        if show_preview:
                            p1 = lms_xy[LEFT_OUTER_EYE]
                            p2 = lms_xy[RIGHT_OUTER_EYE]
                            x1, y1 = int(p1[0]), int(p1[1])
                            x2, y2 = int(p2[0]), int(p2[1])
                            cv2.circle(frame, (x1, y1), 2, (0, 255, 0), -1)
                            cv2.circle(frame, (x2, y2), 2, (0, 255, 0), -1)
                            cv2.putText(frame, f"s(px)={s:.1f}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(frame, f"Collecting... {len(s_values)}", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if show_preview:
                    cv2.imshow("Distance calibration (ESC to stop)", frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()

        if len(s_values) < int(min_samples):
            raise RuntimeError(
                f"Not enough valid samples: {len(s_values)} < {min_samples}. "
                "Try better lighting / face camera / increase duration."
            )

        s0_px = float(np.median(np.asarray(s_values, dtype=np.float32)))
        calib = DistanceCalib(Z0_cm=float(Z0_cm), s0_px=s0_px)
        self.distance_calib = calib
        return calib
