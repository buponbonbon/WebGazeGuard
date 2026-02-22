from __future__ import annotations

import math
from typing import Optional, Dict, Tuple

import cv2
import numpy as np

# Sparse 2D landmark indices (MediaPipe FaceLandmarker / FaceMesh convention)
POSE_LMK_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# Canonical 3D face model points corresponding to POSE_LMK_IDX order.
FACE_3D_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, 63.6, -12.5),       # chin
        (-43.3, -32.7, -26.0),    # left eye outer
        (43.3, -32.7, -26.0),     # right eye outer
        (-28.9, 28.9, -24.1),     # left mouth
        (28.9, 28.9, -24.1),      # right mouth
    ],
    dtype=np.float32,
)

def _camera_matrix(w: int, h: int, focal_scale: float = 1.0) -> np.ndarray:
    f = float(focal_scale) * float(max(w, h))
    cx, cy = w / 2.0, h / 2.0
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

def _rotation_matrix_to_euler_ypr(R: np.ndarray) -> Tuple[float, float, float]:
    # yaw(y), pitch(x), roll(z) in degrees
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0

    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))

def estimate_head_pose_pnp(
    lms_xy: Dict[int, Tuple[float, float]],
    frame_shape,
    focal_scale: float = 1.0,
) -> Optional[Dict[str, object]]:
    """Estimate head pose (yaw, pitch, roll) from 2D landmarks using solvePnP."""
    h, w = frame_shape[:2]
    K = _camera_matrix(w, h, focal_scale=focal_scale)
    dist = np.zeros((4, 1), dtype=np.float32)

    try:
        image_points = np.array(
            [
                lms_xy[POSE_LMK_IDX["nose_tip"]],
                lms_xy[POSE_LMK_IDX["chin"]],
                lms_xy[POSE_LMK_IDX["left_eye_outer"]],
                lms_xy[POSE_LMK_IDX["right_eye_outer"]],
                lms_xy[POSE_LMK_IDX["left_mouth"]],
                lms_xy[POSE_LMK_IDX["right_mouth"]],
            ],
            dtype=np.float32,
        )
    except KeyError:
        return None

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=FACE_3D_MODEL,
        imagePoints=image_points,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _rotation_matrix_to_euler_ypr(R)

    return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll), "rvec": rvec, "tvec": tvec}
