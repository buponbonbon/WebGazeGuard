from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

from core.schemas import CVFeatures
from vision.landmarks import FaceLandmarker, FaceLandmarks
from vision.ear import compute_ear_both_eyes
from vision.blink import BlinkDetector
from vision.head_pose import estimate_head_pose_pnp

# MediaPipe Face Mesh indices (stable across versions)
LEFT_OUTER_EYE = 33
RIGHT_OUTER_EYE = 263

# Small sets around the eyelids/corners for a robust eye ROI box
LEFT_EYE_IDXS = [33, 133, 160, 159, 158, 157, 173]
RIGHT_EYE_IDXS = [362, 263, 387, 386, 385, 384, 398]


@dataclass(frozen=True)
class DistanceCalib:
    # Calibration: Z = Z0 * (s0 / s)
    Z0_cm: float
    s0_px: float


def _xy_dict_to_array(xy: dict[int, Tuple[float, float]], n_points: int = 468) -> np.ndarray:
    """Convert {idx:(x,y)} dict to (n_points,2) array."""
    arr = np.zeros((n_points, 2), dtype=np.float32)
    for i, (x, y) in xy.items():
        if 0 <= i < n_points:
            arr[i, 0] = float(x)
            arr[i, 1] = float(y)
    return arr


def _interocular_px(lms_xy: np.ndarray) -> float:
    """Inter-ocular pixel distance (33 ↔ 263)."""
    if lms_xy is None or lms_xy.shape[0] <= RIGHT_OUTER_EYE:
        return float("nan")
    return float(np.linalg.norm(lms_xy[LEFT_OUTER_EYE] - lms_xy[RIGHT_OUTER_EYE]))


def _estimate_distance_cm(lms_xy: np.ndarray, calib: DistanceCalib) -> Optional[float]:
    """Estimate viewing distance (cm) using a simple 1-point calibration."""
    s = _interocular_px(lms_xy)
    if not np.isfinite(s) or s <= 0:
        return None
    if calib.Z0_cm <= 0 or calib.s0_px <= 0:
        return None
    return float(calib.Z0_cm * (calib.s0_px / s))


def _categorize_distance(distance_cm: Optional[float], near_cm: float, far_cm: float) -> Optional[str]:
    """Map a distance to {too_close, normal, too_far}."""
    if distance_cm is None or not np.isfinite(distance_cm):
        return None
    if distance_cm < near_cm:
        return "too_close"
    if distance_cm > far_cm:
        return "too_far"
    return "normal"


def _clip_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def _crop_eye_roi(frame_bgr: np.ndarray, lms_xy: np.ndarray, margin: float = 0.25) -> Optional[np.ndarray]:
    """Crop an eye ROI (both eyes) from the frame using landmark bbox."""
    if lms_xy is None or lms_xy.shape[0] < 399:
        return None

    H, W = frame_bgr.shape[:2]

    idxs = LEFT_EYE_IDXS + RIGHT_EYE_IDXS
    pts = lms_xy[idxs]  # (N,2) absolute pixel coords

    x0, y0 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x1, y1 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))

    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)

    # Expand bbox a bit to include eyelids/eye corners
    x0 -= margin * bw
    x1 += margin * bw
    y0 -= margin * bh
    y1 += margin * bh

    x0i = _clip_int(x0, 0, W - 1)
    x1i = _clip_int(x1, 0, W - 1)
    y0i = _clip_int(y0, 0, H - 1)
    y1i = _clip_int(y1, 0, H - 1)

    if x1i <= x0i or y1i <= y0i:
        return None

    return frame_bgr[y0i:y1i, x0i:x1i].copy()


class VisionExtractor:
    """Frame-level CV feature extractor (landmarks → EAR/blink/pose/distance [+ optional gaze])."""

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
        # Optional gaze CNN (if provided, we infer yaw/pitch from eye ROI)
        gaze_ckpt_path: Optional[str] = None,
        gaze_model_name: str = "custom",
        gaze_input_channels: int = 1,
        gaze_output_dim: int = 2,
        gaze_resize_hw: Tuple[int, int] = (64, 64),
        gaze_every_n_frames: int = 3,  # run gaze model every N frames to keep FPS
    ):
        self.landmarker = FaceLandmarker(model_path=landmark_model_path)
        self.blink = BlinkDetector(ear_thresh=ear_thresh, consec_frames=ear_consec_frames)
        self.focal_scale = float(focal_scale)

        self.distance_calib = distance_calib
        self.distance_near_cm = float(distance_near_cm)
        self.distance_far_cm = float(distance_far_cm)

        # Pose history for blink freeze during fast head turns
        self._prev_pose_ts_ms: Optional[int] = None
        self._prev_yaw: Optional[float] = None
        self._prev_pitch: Optional[float] = None
        self._freeze_blink_until_ms: Optional[int] = None

        # Gaze inference (optional)
        self._gaze_model = None
        self._gaze_cfg = dict(
            input_channels=int(gaze_input_channels),
            output_dim=int(gaze_output_dim),
            model_name=str(gaze_model_name),
            resize_hw=tuple(gaze_resize_hw),
        )
        self._gaze_every_n = max(1, int(gaze_every_n_frames))
        self._frame_counter = 0
        self._last_gaze: Optional[np.ndarray] = None

        if gaze_ckpt_path:
            # Import here to avoid importing torch in pure CV mode
            from ml_cnn.infer import load_model  # your NEW infer.py

            self._gaze_model = load_model(
                gaze_ckpt_path,
                input_channels=self._gaze_cfg["input_channels"],
                output_dim=self._gaze_cfg["output_dim"],
                model_name=self._gaze_cfg["model_name"],
                pretrained=False,
            )

    def _maybe_predict_gaze(self, frame_bgr: np.ndarray, lms_xy: np.ndarray) -> Optional[np.ndarray]:
        """Return last gaze prediction (yaw,pitch). Runs model every N frames."""
        if self._gaze_model is None:
            return None

        self._frame_counter += 1
        if (self._frame_counter % self._gaze_every_n) != 0 and self._last_gaze is not None:
            return self._last_gaze

        roi = _crop_eye_roi(frame_bgr, lms_xy)
        if roi is None:
            return self._last_gaze

        from ml_cnn.infer import predict_array  # your NEW infer.py

        gaze = predict_array(
            self._gaze_model,
            roi,
            input_channels=self._gaze_cfg["input_channels"],
            resize_hw=self._gaze_cfg["resize_hw"],
        )
        self._last_gaze = np.asarray(gaze, dtype=np.float32).reshape(-1)
        return self._last_gaze

    def extract(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> CVFeatures:
        """Extract CVFeatures from a single BGR frame."""
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

        # EAR (left/right/mean)
        ear_l, ear_r, ear_m = compute_ear_both_eyes(lms_xy)

        # Head pose (yaw/pitch/roll)
        pose = estimate_head_pose_pnp(lms_xy, frame_bgr.shape, focal_scale=self.focal_scale)
        yaw = pose["yaw"] if pose else None
        pitch = pose["pitch"] if pose else None
        roll = pose["roll"] if pose else None

        # Blink (freeze briefly during fast head motion)
        blink_flag = False
        freeze_active = self._freeze_blink_until_ms is not None and timestamp_ms < self._freeze_blink_until_ms

        if not freeze_active and yaw is not None and pitch is not None:
            if self._prev_pose_ts_ms is not None:
                dt = (timestamp_ms - self._prev_pose_ts_ms) / 1000.0
                if dt > 0 and self._prev_yaw is not None and self._prev_pitch is not None:
                    yaw_vel = abs(yaw - self._prev_yaw) / dt
                    pitch_vel = abs(pitch - self._prev_pitch) / dt

                    if yaw_vel > 150 or pitch_vel > 150:
                        self._freeze_blink_until_ms = timestamp_ms + 250
                        self.blink.update(1.0)  # reset detector state
                    else:
                        blink_flag = self.blink.update(ear_m)
                else:
                    blink_flag = self.blink.update(ear_m)
            else:
                blink_flag = self.blink.update(ear_m)
        else:
            self.blink.update(1.0)

        self._prev_pose_ts_ms = timestamp_ms
        self._prev_yaw = yaw
        self._prev_pitch = pitch

        # Distance (optional calibration)
        distance_cm: Optional[float] = None
        distance_cat: Optional[str] = None
        if self.distance_calib is not None:
            distance_cm = _estimate_distance_cm(lms_xy, self.distance_calib)
            distance_cat = _categorize_distance(distance_cm, self.distance_near_cm, self.distance_far_cm)

        # Optional gaze prediction (yaw,pitch from eye ROI)
        gaze = self._maybe_predict_gaze(frame_bgr, lms_xy)

        # Build payload; keep backward-compat with older CVFeatures schemas.
        payload: Dict[str, Any] = dict(
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

        # Only include gaze fields if CVFeatures schema supports them.
        if gaze is not None and gaze.size >= 2:
            payload["gaze_yaw"] = float(gaze[0])
            payload["gaze_pitch"] = float(gaze[1])

        try:
            return CVFeatures(**payload)
        except TypeError:
            # Schema doesn't have gaze_* fields (older core.schemas). Remove and retry.
            payload.pop("gaze_yaw", None)
            payload.pop("gaze_pitch", None)
            return CVFeatures(**payload)
