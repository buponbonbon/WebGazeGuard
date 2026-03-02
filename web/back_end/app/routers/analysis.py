from __future__ import annotations
import time
import json
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
import io
import csv

from ..deps import get_current_user_id
from ..schemas import FrameAnalysisOut, Metrics
from ..services.cv_adapter import analyze_frame
from ..services.risk_engine import compute_risk

router = APIRouter()

# In-memory per-user session metrics history (for demo). For production use Redis.
_history: Dict[int, List[dict]] = {}

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
            ts_ms = int(data.get("ts_ms") or (time.time()*1000))
            jpeg_b64 = data.get("jpeg_b64")
            if not jpeg_b64:
                continue

            cv = await analyze_frame(user_id, jpeg_b64)
            risk = compute_risk(cv.blink_rate_per_min, cv.yaw_deg, cv.pitch_deg, cv.distance_cm)

            out = FrameAnalysisOut(
                ts_ms=ts_ms,
                metrics=Metrics(
                    blink_rate_per_min=cv.blink_rate_per_min,
                    ear=cv.ear,
                    head_pose_yaw_deg=cv.yaw_deg,
                    head_pose_pitch_deg=cv.pitch_deg,
                    distance_cm=cv.distance_cm,
                    strain_risk=risk.strain_risk,
                    posture_flag=risk.posture_flag,
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
    writer.writerow(["ts_ms","blink_rate_per_min","ear","yaw_deg","pitch_deg","distance_cm","strain_risk","posture_flag"])
    for r in rows:
        m = r["metrics"]
        writer.writerow([
            r["ts_ms"], m["blink_rate_per_min"], m["ear"],
            m["head_pose_yaw_deg"], m["head_pose_pitch_deg"],
            m["distance_cm"], m["strain_risk"], m.get("posture_flag","")
        ])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=session_metrics.csv"})
