from __future__ import annotations
import time
import json
import base64
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional
import io
import csv

import numpy as np
import cv2

from ..deps import get_current_user_id
from ..schemas import FrameAnalysisOut, Metrics

# WebGazeGuard pipeline + fusion
from core.pipeline import create_state, step
from fusion.risk_engine import assess_risk

router = APIRouter()

# In-memory per-user session metrics history (for demo). For production use Redis.
_history: Dict[int, List[dict]] = {}


def _decode_jpeg_b64(jpeg_b64: str) -> Optional[np.ndarray]:
    """Decode base64 JPEG into BGR image (np.ndarray). Accepts optional data URL prefix."""
    if not jpeg_b64:
        return None
    # strip data URL prefix if any: data:image/jpeg;base64,...
    if "," in jpeg_b64 and jpeg_b64.strip().lower().startswith("data:"):
        jpeg_b64 = jpeg_b64.split(",", 1)[1]
    try:
        data = base64.b64decode(jpeg_b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


@router.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """WebSocket for real-time frames.

    Client must send:
    - First message: {"type":"auth","token":"..."}
    - Then frames: {"type":"frame","ts_ms":..., "jpeg_b64":"..."}
    Server replies:
    - {"type":"metrics","payload":{FrameAnalysisOut}}
    """
    await ws.accept()
    user_id = None

    # Create per-connection pipeline state (temporal windowing lives here)
    state = create_state()

    try:
        first = await ws.receive_text()
        obj = json.loads(first)
        if obj.get("type") != "auth" or not obj.get("token"):
            await ws.close(code=1008)
            return

        # Reuse auth dependency logic lightly here
        from ..utils.jwt import decode_token
        payload = decode_token(obj["token"])
        if not payload or "sub" not in payload:
            await ws.close(code=1008)
            return
        user_id = int(payload["sub"])

        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            if data.get("type") != "frame":
                continue

            ts_ms = int(data.get("ts_ms") or (time.time() * 1000))
            jpeg_b64 = data.get("jpeg_b64")
            if not jpeg_b64:
                continue

            frame_bgr = _decode_jpeg_b64(jpeg_b64)
            if frame_bgr is None:
                continue

            # Run one pipeline step: extract CV features + update temporal window
            cvf, win = step(state, frame_bgr, timestamp_ms=ts_ms)

            # Optional text (if client sends it later)
            text = data.get("text")  # can be None

            # Assess risk using fusion engine (NLP optional)
            risk = assess_risk(win, nlp=None, weights=None)

            # Best-effort fields from CVFeatures
            blink_rate = float(getattr(cvf, "blink_rate_per_min", 0.0) or 0.0)
            ear = float(getattr(cvf, "ear", 0.0) or 0.0)
            yaw = float(getattr(cvf, "yaw_deg", getattr(cvf, "head_pose_yaw_deg", 0.0)) or 0.0)
            pitch = float(getattr(cvf, "pitch_deg", getattr(cvf, "head_pose_pitch_deg", 0.0)) or 0.0)
            dist = float(getattr(cvf, "distance_cm", 0.0) or 0.0)

            out = FrameAnalysisOut(
                ts_ms=ts_ms,
                metrics=Metrics(
                    blink_rate_per_min=blink_rate,
                    ear=ear,
                    head_pose_yaw_deg=yaw,
                    head_pose_pitch_deg=pitch,
                    distance_cm=dist,
                    strain_risk=float(getattr(risk, "risk_score", 0.0) or 0.0),
                    posture_flag=getattr(risk, "posture_flag", None),
                ),
            )

            _history.setdefault(user_id, []).append(out.model_dump())
            _history[user_id] = _history[user_id][-1200:]  # keep last ~1200 frames

            await ws.send_text(json.dumps({"type": "metrics", "payload": out.model_dump()}))

    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await ws.close(code=1011)
        except Exception:
            pass


@router.get("/export.csv")
def export_csv(user_id: int = Depends(get_current_user_id)):
    """Download last session metrics as CSV."""
    rows = _history.get(user_id, [])
    if not rows:
        raise HTTPException(status_code=404, detail="No session data yet")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "ts_ms",
        "blink_rate_per_min",
        "ear",
        "yaw_deg",
        "pitch_deg",
        "distance_cm",
        "strain_risk",
        "posture_flag",
    ])
    for r in rows:
        m = r["metrics"]
        writer.writerow([
            r["ts_ms"],
            m["blink_rate_per_min"],
            m["ear"],
            m["head_pose_yaw_deg"],
            m["head_pose_pitch_deg"],
            m["distance_cm"],
            m["strain_risk"],
            m.get("posture_flag", ""),
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=session_metrics.csv"},
    )
