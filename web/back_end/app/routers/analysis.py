# analysis.py (WebSocket streaming router) - patched for stable realtime UI
# - fixes warm-up where temporal window can be None
# - robustly maps metric field names across CV/temporal modules
# - avoids UI spam by using a neutral distance when distance is not yet calibrated

from __future__ import annotations

import time
import json
import base64
import traceback
import io
import csv
from typing import Dict, List, Optional

import numpy as np
import cv2
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse

from ..deps import get_current_user_id
from ..schemas import FrameAnalysisOut, Metrics

from core.config import Config
from core.pipeline import build_pipeline, step
from fusion.risk_engine import assess_risk

router = APIRouter()

_history: Dict[int, List[dict]] = {}


def _decode_jpeg_b64(jpeg_b64: str) -> Optional[np.ndarray]:
    if not jpeg_b64:
        return None
    # support "data:image/jpeg;base64,...."
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
    await ws.accept()
    state = build_pipeline(Config(), gaze_ckpt_path=None)

    try:
        # 1) auth first message
        first = await ws.receive_text()
        obj = json.loads(first)

        if obj.get("type") != "auth" or not obj.get("token"):
            await ws.close(code=1008)
            return

        from ..utils.jwt import decode_token
        payload = decode_token(obj["token"])
        if not payload or "sub" not in payload:
            await ws.close(code=1008)
            return

        user_id = int(payload["sub"])

        # Keep last valid temporal window (warm-up frames may return win=None)
        last_valid_win = None

        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            if data.get("type") != "frame":
                # ignore unknown message types
                continue

            ts_ms = int(data.get("ts_ms") or (time.time() * 1000))
            jpeg_b64 = data.get("jpeg_b64")

            frame_bgr = _decode_jpeg_b64(jpeg_b64)
            if frame_bgr is None:
                continue

            try:
                cvf, win = step(state, frame_bgr, timestamp_ms=ts_ms)

                if win is not None:
                    last_valid_win = win
                win_for_risk = last_valid_win

                # risk may be skipped during warm-up
                risk = assess_risk(win_for_risk, nlp=None, weights=None) if win_for_risk is not None else None

                # ---- robust metric mapping (handles differing attribute names) ----
                # Blink rate: prefer temporal window if available
                blink_rate = None
                if win_for_risk is not None:
                    blink_rate = getattr(win_for_risk, "blink_rate_bpm", None)
                if blink_rate is None:
                    blink_rate = getattr(cvf, "blink_rate_bpm", None)
                if blink_rate is None:
                    blink_rate = getattr(cvf, "blink_rate_per_min", None)
                blink_rate = float(blink_rate or 0.0)

                # EAR: various naming across modules
                ear = getattr(cvf, "ear", None)
                if ear is None:
                    ear = getattr(cvf, "ear_mean", None)
                if ear is None:
                    ear = getattr(cvf, "ear_avg", None)
                ear = float(ear or 0.0)

                # Head pose
                yaw = getattr(cvf, "head_pose_yaw_deg", None)
                if yaw is None:
                    yaw = getattr(cvf, "yaw_deg", None)
                yaw = float(yaw or 0.0)

                pitch = getattr(cvf, "head_pose_pitch_deg", None)
                if pitch is None:
                    pitch = getattr(cvf, "pitch_deg", None)
                pitch = float(pitch or 0.0)

                # Distance: may be 0 until calibrated; avoid spamming 'too close' on UI
                dist = getattr(cvf, "distance_cm", None)
                if dist is None:
                    dist = getattr(cvf, "head_distance_cm", None)
                dist = float(dist or 0.0)
                if dist <= 0.0:
                    # Neutral fallback for UI so it doesn't scream "too close" during warm-up/un-calibrated state.
                    dist = 60.0

                out = FrameAnalysisOut(
                    ts_ms=ts_ms,
                    metrics=Metrics(
                        blink_rate_per_min=blink_rate,
                        ear=ear,
                        head_pose_yaw_deg=yaw,
                        head_pose_pitch_deg=pitch,
                        distance_cm=dist,
                        strain_risk=float(getattr(risk, "risk_score", 0.0) or 0.0) if risk is not None else 0.0,
                        posture_flag=getattr(risk, "posture_flag", None) if risk is not None else None,
                    ),
                )

                _history.setdefault(user_id, []).append(out.model_dump())
                _history[user_id] = _history[user_id][-1200:]  # cap

                await ws.send_text(json.dumps({"type": "metrics", "payload": out.model_dump()}))

            except Exception as e:
                print("🔥 WebSocket frame processing error:")
                traceback.print_exc()
                await ws.send_text(json.dumps({"type": "error", "message": str(e)}))

    except WebSocketDisconnect:
        return
    except Exception:
        print("🔥 Fatal WebSocket error:")
        traceback.print_exc()
        try:
            await ws.close(code=1011)
        except Exception:
            pass


@router.get("/export.csv")
def export_csv(user_id: int = Depends(get_current_user_id)):
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
