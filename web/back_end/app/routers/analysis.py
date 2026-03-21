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
from vision.head_distance import calibrate_from_frames, DistanceCalib
from fusion.risk_engine import assess_risk

router = APIRouter()

_history: Dict[int, List[dict]] = {}


def _decode_jpeg_b64(jpeg_b64: str) -> Optional[np.ndarray]:
    if not jpeg_b64:
        return None

    if "," in jpeg_b64 and jpeg_b64.strip().lower().startswith("data:"):
        jpeg_b64 = jpeg_b64.split(",", 1)[1]
    try:
        data = base64.b64decode(jpeg_b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


async def _safe_send_text(ws: WebSocket, payload: dict) -> bool:
    # Send JSON payload over WS. Return False if client disconnected.
    try:
        await ws.send_text(json.dumps(payload))
        return True
    except WebSocketDisconnect:
        return False
    except Exception:
        # any other send error -> treat as disconnected
        return False


@router.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    cfg = Config()
    state = build_pipeline(cfg, gaze_ckpt_path=getattr(cfg, "gaze_ckpt_path", None))

    try:
        # DEMO MODE: still consume first message, but skip token validation
        first = await ws.receive_text()
        obj = json.loads(first)
        user_id = 1

        # Keep last valid temporal window (warm-up frames may return win=None)
        last_valid_win = None

        # Distance calibration via WS:
        # Client sends: {type:'calibrate', Z0_cm:60, n_frames:20}
        calib_collect = None  # dict with keys: Z0_cm, n_frames, s_values(list)

        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            msg_type = data.get("type")
            if msg_type == "calibrate":
                # start collecting interocular pixel distances from next frames
                Z0_cm = float(data.get("Z0_cm") or 60.0)
                n_frames = int(data.get("n_frames") or 20)
                n_frames = max(5, min(120, n_frames))
                calib_collect = {"Z0_cm": Z0_cm, "n_frames": n_frames, "s_values": []}
                ok = await _safe_send_text(
                    ws,
                    {
                        "type": "toast",
                        "level": "info",
                        "message": f"Đang hiệu chuẩn ({n_frames} frames). Giữ mặt ổn định ở ~{Z0_cm:.0f}cm",
                    },
                )
                if not ok:
                    return
                continue

            if msg_type != "frame":
                # ignore unknown message types
                continue

            ts_ms = int(data.get("ts_ms") or (time.time() * 1000))
            jpeg_b64 = data.get("jpeg_b64")

            frame_bgr = _decode_jpeg_b64(jpeg_b64)
            if frame_bgr is None:
                continue

            try:
                cvf, win = step(state, frame_bgr, timestamp_ms=ts_ms)

                # --- calibration collection ---
                if calib_collect is not None:
                    s_px = getattr(state.vision, "_last_s_px", None)
                    if s_px is not None and np.isfinite(s_px) and float(s_px) > 0:
                        calib_collect["s_values"].append(float(s_px))

                    if len(calib_collect["s_values"]) >= int(calib_collect["n_frames"]):
                        try:
                            calib = calibrate_from_frames(
                                np.asarray(calib_collect["s_values"], dtype=np.float64),
                                Z0_cm=float(calib_collect["Z0_cm"]),
                            )
                            # update live pipeline calibration
                            state.vision.distance_calib = calib

                            ok = await _safe_send_text(
                                ws,
                                {
                                    "type": "calibrated",
                                    "payload": {"Z0_cm": calib.Z0_cm, "s0_px": calib.s0_px},
                                },
                            )
                            if not ok:
                                return

                            ok = await _safe_send_text(
                                ws,
                                {
                                    "type": "toast",
                                    "level": "ok",
                                    "message": f"Hiệu chuẩn xong: Z0={calib.Z0_cm:.0f}cm, s0={calib.s0_px:.1f}px",
                                },
                            )
                            if not ok:
                                return
                        except Exception:
                            ok = await _safe_send_text(
                                ws,
                                {
                                    "type": "toast",
                                    "level": "info",
                                    "message": "Hiệu chuẩn thất bại, vui lòng thử lại.",
                                },
                            )
                            if not ok:
                                return
                        calib_collect = None

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

                # Head pose (core.schemas.CVFeatures uses yaw/pitch/roll)
                yaw = getattr(cvf, "head_pose_yaw_deg", None)
                if yaw is None:
                    yaw = getattr(cvf, "yaw_deg", None)
                if yaw is None:
                    yaw = getattr(cvf, "yaw", None)
                yaw = float(yaw or 0.0)

                pitch = getattr(cvf, "head_pose_pitch_deg", None)
                if pitch is None:
                    pitch = getattr(cvf, "pitch_deg", None)
                if pitch is None:
                    pitch = getattr(cvf, "pitch", None)
                pitch = float(pitch or 0.0)

                # Distance: may be 0 until calibrated; avoid spamming 'too close' on UI
                dist = getattr(cvf, "distance_cm", None)
                if dist is None:
                    dist = getattr(cvf, "head_distance_cm", None)
                dist = float(dist or 0.0)
                if dist <= 0.0:
                    # Neutral fallback for UI so it doesn't scream "too close" during warm-up/un-calibrated state.
                    dist = 60.0

                gaze_yaw = getattr(cvf, "gaze_yaw", None)
                gaze_pitch = getattr(cvf, "gaze_pitch", None)

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
                        gaze_yaw_deg=float(gaze_yaw) if gaze_yaw is not None else None,
                        gaze_pitch_deg=float(gaze_pitch) if gaze_pitch is not None else None,
                    ),
                )

                _history.setdefault(user_id, []).append(out.model_dump())
                _history[user_id] = _history[user_id][-1200:]  # cap

                ok = await _safe_send_text(ws, {"type": "metrics", "payload": out.model_dump()})
                if not ok:
                    return

            except Exception as e:
                print("🔥 WebSocket frame processing error:")
                traceback.print_exc()
                ok = await _safe_send_text(ws, {"type": "error", "message": str(e)})
                if not ok:
                    return

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