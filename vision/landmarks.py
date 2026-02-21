from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


@dataclass
class FaceLandmarks:
    # pixel coordinates (x_px, y_px) for 468 points
    xy: dict[int, Tuple[float, float]]
    # optional z (relative) if you need it later
    z: Optional[np.ndarray] = None
    confidence: Optional[float] = None


def ensure_face_landmarker_model(model_path: str = "face_landmarker.task", url: str = DEFAULT_MODEL_URL) -> str:
    """Download the MediaPipe face landmarker model if missing."""
    if os.path.exists(model_path):
        return model_path

    import urllib.request

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, model_path)
    return model_path


class FaceLandmarker:
    """Thin wrapper to get 468 landmarks from a single RGB/BGR frame."""

    def __init__(self, model_path: str = "face_landmarker.task"):
        model_path = ensure_face_landmarker_model(model_path=model_path)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect_xy(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarks]:
        """Return pixel-space (x,y) dict for 468 landmarks, or None if no face."""
        if frame_bgr is None:
            return None

        # MediaPipe Tasks expects RGB
        frame_rgb = frame_bgr[..., ::-1].copy()
        h, w = frame_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        lms = result.face_landmarks[0]  # 468 landmarks
        xy = {}
        z = np.zeros((len(lms),), dtype=np.float32)

        for i, lm in enumerate(lms):
            xy[i] = (float(lm.x * w), float(lm.y * h))
            z[i] = float(lm.z)

        # Return None when face is not detected to avoid breaking the pipeline
        return FaceLandmarks(xy=xy, z=z, confidence=None)
