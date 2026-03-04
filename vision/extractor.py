
"""
vision/extractor.py (eye-ROI tuned)
- Keeps VisionExtractor for pipeline import.
- Uses true eye ROI (tight) to reduce head-bias.
- Short English comments.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from core.schemas import CVFeatures
from vision.landmarks import FaceLandmarker
from vision.ear import compute_ear_both_eyes
from vision.blink import BlinkDetector
from vision.head_pose import estimate_head_pose_pnp
from vision.head_distance import estimate_distance_cm, DistanceCalib

# Face Mesh indices
L_EYE_OUTER = 33
R_EYE_OUTER = 263
LEFT_EYE_IDXS = [33, 133, 160, 159, 158, 157, 173]
RIGHT_EYE_IDXS = [362, 263, 387, 386, 385, 384, 398]


def _dict_to_array(xy_dict: Dict[int, Tuple[float, float]], n: int = 468) -> np.ndarray:
    """Convert {idx:(x,y)} -> (n,2) array; missing points become (0,0)."""
    arr = np.zeros((n, 2), dtype=np.float32)
    for i, (x, y) in xy_dict.items():
        if 0 <= i < n:
            arr[i, 0] = float(x)
            arr[i, 1] = float(y)
    return arr


def _valid_pt(p: np.ndarray) -> bool:
    return bool(np.isfinite(p).all() and not (p[0] == 0.0 and p[1] == 0.0))


def _clip(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(v)))))


def _interocular_px(lms_xy: np.ndarray) -> float:
    """s = ||p33 - p263|| in pixels."""
    if lms_xy is None or lms_xy.shape[0] <= R_EYE_OUTER:
        return float("nan")
    pL = lms_xy[L_EYE_OUTER]
    pR = lms_xy[R_EYE_OUTER]
    if not (_valid_pt(pL) and _valid_pt(pR)):
        return float("nan")
    return float(np.linalg.norm(pL - pR))


def _crop_bbox_from_idxs(
    frame_bgr: np.ndarray,
    lms_xy: np.ndarray,
    idxs: List[int],
    *,
    margin: float,
    min_w: int,
    min_h: int,
) -> Optional[np.ndarray]:
    """Crop ROI by bbox of selected landmarks (tight + padded)."""
    H, W = frame_bgr.shape[:2]
    pts = []
    for i in idxs:
        if 0 <= i < lms_xy.shape[0]:
            p = lms_xy[i]
            if _valid_pt(p):
                pts.append(p)
    if not pts:
        return None

    pts = np.stack(pts, axis=0)
    x0, y0 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x1, y1 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))

    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)

    x0 -= margin * bw
    x1 += margin * bw
    y0 -= margin * bh
    y1 += margin * bh

    # Enforce minimum size
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    bw2 = max(float(min_w), x1 - x0)
    bh2 = max(float(min_h), y1 - y0)
    x0 = cx - bw2 / 2.0
    x1 = cx + bw2 / 2.0
    y0 = cy - bh2 / 2.0
    y1 = cy + bh2 / 2.0

    x0i = _clip(x0, 0, W - 1)
    x1i = _clip(x1, 0, W - 1)
    y0i = _clip(y0, 0, H - 1)
    y1i = _clip(y1, 0, H - 1)

    if x1i <= x0i or y1i <= y0i:
        return None

    roi = frame_bgr[y0i:y1i, x0i:x1i]
    if roi.size == 0:
        return None
    print("ROI shape:", roi.shape)
    return roi.copy()


class VisionExtractor:
    """
    Frame-level CV extractor used by core.pipeline:
      landmarks -> EAR/blink -> head pose
      + optional gaze CNN yaw/pitch (tight eye ROI)
    """

    def __init__(
        self,
        *,
        ear_thresh: float = 0.20,
        ear_consec_frames: int = 2,
        distance_calib=None,              # kept for compatibility
        gaze_ckpt_path: Optional[str] = None,
        gaze_every_n_frames: int = 3,
        gaze_resize_hw: Tuple[int, int] = (64, 64),
    ):
        self.landmarker = FaceLandmarker()
        self.blink = BlinkDetector(ear_thresh, ear_consec_frames)

        self.distance_calib = distance_calib
        self._last_s_px: Optional[float] = None

        self.gaze_ckpt_path = gaze_ckpt_path
        self.gaze_every_n = max(1, int(gaze_every_n_frames))
        self.gaze_resize_hw = tuple(gaze_resize_hw)

        self._gaze_model = None
        self._gaze_i = 0
        self._last_gaze: Optional[np.ndarray] = None

    def _ensure_gaze_model(self):
        if self.gaze_ckpt_path is None:
            return None
        if self._gaze_model is not None:
            return self._gaze_model
        from ml_cnn.infer import load_model
        self._gaze_model = load_model(self.gaze_ckpt_path)
        return self._gaze_model

    def _predict_gaze(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Predict yaw/pitch from ROI."""
        model = self._ensure_gaze_model()
        if model is None:
            return None

        # Preferred helper
        try:
            from ml_cnn.infer import predict_array  # type: ignore
            pred = predict_array(model, roi_bgr, resize_hw=self.gaze_resize_hw)
            arr = np.asarray(pred, dtype=np.float32).reshape(-1)
            if arr.size >= 2 and np.isfinite(arr[:2]).all():
                return arr[:2]
        except Exception:
            pass

        # Fallback minimal
        try:
            import cv2
            import torch

            H, W = self.gaze_resize_hw
            x = cv2.resize(roi_bgr, (W, H), interpolation=cv2.INTER_AREA)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            x = x[None, None, :, :]

            device = next(model.parameters()).device
            t = torch.from_numpy(x).to(device)
            with torch.no_grad():
                out = model(t).detach().cpu().numpy()[0]
            out = np.asarray(out, dtype=np.float32).reshape(-1)
            if out.size >= 2 and np.isfinite(out[:2]).all():
                return out[:2]
        except Exception:
            return None

        return None

    def _crop_eye_roi(self, frame_bgr: np.ndarray, lms_xy: np.ndarray) -> Optional[np.ndarray]:
        """
        Tight eye ROI:
        - bbox on eyelid landmarks (not midpoint).
        - bounded size to reduce head bias.
        """
        if lms_xy is None or lms_xy.shape[0] < 400:
            return None

        s = _interocular_px(lms_xy)
        if not np.isfinite(s) or s <= 1:
            return None

        min_w = int(max(40, min(80, s * 0.35)))
        min_h = int(max(24, min(55, s * 0.25)))

        roi = _crop_bbox_from_idxs(frame_bgr, lms_xy, LEFT_EYE_IDXS, margin=0.55, min_w=min_w, min_h=min_h)
        if roi is not None:
            return roi
        roi = _crop_bbox_from_idxs(frame_bgr, lms_xy, RIGHT_EYE_IDXS, margin=0.55, min_w=min_w, min_h=min_h)
        if roi is not None:
            return roi
        return _crop_bbox_from_idxs(
            frame_bgr,
            lms_xy,
            LEFT_EYE_IDXS + RIGHT_EYE_IDXS,
            margin=0.40,
            min_w=int(min_w * 1.4),
            min_h=int(min_h * 1.2),
        )

    def extract(self, frame_bgr, timestamp_ms: int) -> CVFeatures:
        lm = self.landmarker.detect_xy(frame_bgr)

        if lm is None:
            return CVFeatures(
                timestamp_ms=timestamp_ms,
                face_detected=False,
                blink=False,
                total_blinks=self.blink.total_blinks,
            )

        lms_xy = _dict_to_array(lm.xy)

        # --- distance estimation (Z = Z0 * s0 / s) ---
        s_px = _interocular_px(lms_xy)
        self._last_s_px = float(s_px) if np.isfinite(s_px) else None
        distance_cm = estimate_distance_cm(s_px, self.distance_calib) if self.distance_calib is not None else None

        ear_l, ear_r, ear_m = compute_ear_both_eyes(lms_xy)

        pose = estimate_head_pose_pnp(lms_xy, frame_bgr.shape)
        yaw = pose["yaw"] if pose else None
        pitch = pose["pitch"] if pose else None
        roll = pose["roll"] if pose else None

        blink_flag = self.blink.update(float(ear_m))

        payload: Dict[str, Any] = dict(
            timestamp_ms=int(timestamp_ms),
            face_detected=True,
            distance_cm=distance_cm,
            ear_left=float(ear_l),
            ear_right=float(ear_r),
            ear_mean=float(ear_m),
            blink=bool(blink_flag),
            total_blinks=int(self.blink.total_blinks),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )

        if self.gaze_ckpt_path is not None:
            self._gaze_i += 1
            if (self._gaze_i % self.gaze_every_n) != 0 and self._last_gaze is not None:
                gaze = self._last_gaze
            else:
                roi = self._crop_eye_roi(frame_bgr, lms_xy)
                gaze = self._predict_gaze(roi) if roi is not None else self._last_gaze
                if gaze is not None:
                    self._last_gaze = gaze

            if gaze is not None:
                payload["gaze_yaw"] = float(gaze[0])
                payload["gaze_pitch"] = float(gaze[1])

        try:
            return CVFeatures(**payload)
        except TypeError:
            payload.pop("gaze_yaw", None)
            payload.pop("gaze_pitch", None)
            return CVFeatures(**payload)
