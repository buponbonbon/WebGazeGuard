"""CV adapter (placeholder) for your personal AI.
- If you already have your CV pipeline (MediaPipe/ONNX/TensorFlow) as a service,
  set AI_CV_ENDPOINT in .env and we will forward frames to it.
- If not set, we run a LIGHTWEIGHT heuristic analysis on the server so the demo works.

The PDF design mentions 4 core factors: blink, head pose, distance, gaze direction.
In this template we implement a demo version: blink-rate + pseudo EAR + head pose (simulated)
and distance via face width heuristic (simulated) to keep CPU low on common machines.
"""
from __future__ import annotations
import base64
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import cv2
import httpx

from ..settings import settings

@dataclass
class CvResult:
    blink_rate_per_min: float
    ear: float
    yaw_deg: float
    pitch_deg: float
    distance_cm: float

class _BlinkState:
    def __init__(self):
        self.last_blinks: list[float] = []
        self.last_ear: float = 0.25

    def update(self, ear: float) -> float:
        # naive blink detection: ear dip below threshold triggers
        thresh = 0.18
        now = time.time()
        if ear < thresh and self.last_ear >= thresh:
            self.last_blinks.append(now)
        self.last_ear = ear
        # keep last 60 seconds
        self.last_blinks = [t for t in self.last_blinks if now - t <= 60]
        return float(len(self.last_blinks))

_state_by_user: dict[int, _BlinkState] = {}

def _decode_jpeg_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

async def analyze_frame(user_id: int, frame_b64_jpeg: str) -> CvResult:
    # If user has their own CV service, forward to it.
    if settings.ai_cv_endpoint:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.post(settings.ai_cv_endpoint, json={"image_b64": frame_b64_jpeg})
            r.raise_for_status()
            data = r.json()
            return CvResult(
                blink_rate_per_min=float(data.get("blink_rate_per_min", 0.0)),
                ear=float(data.get("ear", 0.0)),
                yaw_deg=float(data.get("yaw_deg", 0.0)),
                pitch_deg=float(data.get("pitch_deg", 0.0)),
                distance_cm=float(data.get("distance_cm", 60.0)),
            )

    # Demo heuristic (fast): compute a simple 'eye openness proxy' using image luminance variance
    img = _decode_jpeg_b64(frame_b64_jpeg)
    if img is None:
        return CvResult(0.0, 0.24, 0.0, 0.0, 60.0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Sample a center patch to estimate "focus/brightness"; map to a pseudo EAR.
    patch = gray[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
    var = float(np.var(patch)) if patch.size else 0.0
    ear = 0.15 + min(0.20, var / 50000.0)  # 0.15..0.35 (rough)

    st = _state_by_user.setdefault(user_id, _BlinkState())
    blinks_last_min = st.update(ear)
    blink_rate = blinks_last_min  # blinks in last 60s ≈ per minute

    # Simulated head pose from image center-of-mass shift (very rough but cheap)
    # If face is off-center, treat as yaw.
    cy, cx = np.array(np.unravel_index(np.argmax(patch), patch.shape)) if patch.size else (0,0)
    yaw = float((cx - patch.shape[1]/2) / max(1.0, patch.shape[1]/2) * 15.0)
    pitch = float((cy - patch.shape[0]/2) / max(1.0, patch.shape[0]/2) * 10.0)

    # Distance demo: use width as inverse proxy (assume 60cm average)
    distance = float(np.clip(60.0 * (640.0 / max(320.0, w)), 35.0, 120.0))

    return CvResult(blink_rate, float(ear), yaw, pitch, distance)
