
from __future__ import annotations

import argparse
import math
import time
from collections import deque

import cv2
import numpy as np

from core.config import Config
from core.pipeline import build_pipeline, step
from vision.landmarks import FaceLandmarker

# Eye outer corners (MediaPipe Face Mesh)
L_EYE_OUTER = 33
R_EYE_OUTER = 263


def now_ms() -> int:
    return int(time.time() * 1000)


def put_text(img, text: str, y: int) -> None:
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)


def _to_rad(a: float) -> float:
    """Auto-detect deg vs rad."""
    return a if abs(a) <= 3.2 else math.radians(a)


def _dict_to_array(xy_dict, n=468) -> np.ndarray:
    arr = np.zeros((n, 2), dtype=np.float32)
    for i, (x, y) in xy_dict.items():
        if 0 <= i < n:
            arr[i] = (float(x), float(y))
    return arr


def _eye_center(frame, lms_xy: np.ndarray) -> tuple[int, int]:
    """Eye-center anchor. Fallback to frame center."""
    h, w = frame.shape[:2]
    if lms_xy is None or lms_xy.shape[0] <= R_EYE_OUTER:
        return w // 2, int(h * 0.42)

    pL = lms_xy[L_EYE_OUTER]
    pR = lms_xy[R_EYE_OUTER]
    if not np.isfinite(pL).all() or not np.isfinite(pR).all():
        return w // 2, int(h * 0.42)
    if (pL[0] == 0 and pL[1] == 0) or (pR[0] == 0 and pR[1] == 0):
        return w // 2, int(h * 0.42)

    cx = int((pL[0] + pR[0]) * 0.5)
    cy = int((pL[1] + pR[1]) * 0.5)
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    return cx, cy


def draw_gaze_arrow(
    frame,
    lms_xy: np.ndarray | None,
    yaw: float | None,
    pitch: float | None,
    *,
    scale_px: int = 180,
    mirror: bool = True,
) -> None:
    """Draw arrow from eye center using yaw/pitch."""
    if yaw is None or pitch is None:
        return
    if not (math.isfinite(yaw) and math.isfinite(pitch)):
        return

    yaw_r = _to_rad(float(yaw))
    pitch_r = _to_rad(float(pitch))

    # Mirror webcam preview: flip yaw
    if mirror:
        yaw_r = -yaw_r

    h, w = frame.shape[:2]
    if lms_xy is None:
        ox, oy = w // 2, int(h * 0.42)
    else:
        ox, oy = _eye_center(frame, lms_xy)

    dx = math.tan(yaw_r)
    dy = -math.tan(pitch_r)

    ex = int(ox + dx * scale_px)
    ey = int(oy + dy * scale_px)

    cv2.arrowedLine(frame, (ox, oy), (ex, ey), (0, 255, 0), 3, tipLength=0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--window", type=float, default=2.5)

    ap.add_argument("--gaze-ckpt", type=str, default="")
    ap.add_argument("--draw-gaze", action="store_true")
    ap.add_argument("--gaze-scale", type=int, default=180)

    ap.add_argument("--ema-alpha", type=float, default=0.55)
    ap.add_argument("--no-smooth", action="store_true")

    ap.add_argument("--mirror", action="store_true", help="Flip yaw for mirrored webcam preview.")

    args = ap.parse_args()

    cfg = Config(fps_assumed=args.fps, window_seconds=args.window)
    state = build_pipeline(cfg, gaze_ckpt_path=args.gaze_ckpt or None)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened")

    # Landmarker for drawing anchor
    landmarker = FaceLandmarker()

    # Calibration (~1s)
    gaze_yaw0 = 0.0
    gaze_pitch0 = 0.0
    has_calib = False
    buf_yaw = deque(maxlen=40)
    buf_pitch = deque(maxlen=40)
    collecting = False
    collect_until = 0

    # Smoothing
    ema_y = None
    ema_p = None
    alpha = float(args.ema_alpha)

    t0 = time.perf_counter()
    frames = 0
    fps_show = 0.0

    print("Controls: Q quit | C calibrate (~1s looking straight) | R reset calibration")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts = now_ms()
        cvf, _win = step(state, frame, timestamp_ms=ts)

        # Landmarks only for arrow anchor
        lm = landmarker.detect_xy(frame)
        lms_xy = _dict_to_array(lm.xy) if lm is not None else None

        gaze_yaw = getattr(cvf, "gaze_yaw", None)
        gaze_pitch = getattr(cvf, "gaze_pitch", None)

        # Collect calibration
        if collecting and gaze_yaw is not None and gaze_pitch is not None:
            buf_yaw.append(float(gaze_yaw))
            buf_pitch.append(float(gaze_pitch))
            if ts >= collect_until:
                if len(buf_yaw) >= 8:
                    gaze_yaw0 = sum(buf_yaw) / len(buf_yaw)
                    gaze_pitch0 = sum(buf_pitch) / len(buf_pitch)
                    has_calib = True
                    ema_y = None
                    ema_p = None
                    print(f"[GAZE] Calibrated yaw0={gaze_yaw0:.4f} pitch0={gaze_pitch0:.4f} (n={len(buf_yaw)})")
                else:
                    print("[GAZE] Calib failed: not enough samples")
                collecting = False

        # Relative gaze
        yaw_rel = None
        pitch_rel = None
        if gaze_yaw is not None and gaze_pitch is not None:
            if has_calib:
                yaw_rel = float(gaze_yaw) - float(gaze_yaw0)
                pitch_rel = float(gaze_pitch) - float(gaze_pitch0)
            else:
                yaw_rel = float(gaze_yaw)
                pitch_rel = float(gaze_pitch)

        yaw_draw = yaw_rel
        pitch_draw = pitch_rel

        if not args.no_smooth and yaw_rel is not None and pitch_rel is not None:
            if ema_y is None:
                ema_y = float(yaw_rel)
                ema_p = float(pitch_rel)
            else:
                ema_y = (1 - alpha) * float(ema_y) + alpha * float(yaw_rel)
                ema_p = (1 - alpha) * float(ema_p) + alpha * float(pitch_rel)
                # auto recenter when near center
                if abs(yaw_rel) < 0.04:
                    ema_y *= 0.85

                if abs(pitch_rel) < 0.04:
                    ema_p *= 0.85
            yaw_draw = float(ema_y)
            pitch_draw = float(ema_p)

        if args.draw_gaze:
            draw_gaze_arrow(
                frame,
                lms_xy,
                yaw_draw,
                pitch_draw,
                scale_px=int(args.gaze_scale),
                mirror=bool(args.mirror),
            )

        # FPS
        frames += 1
        dt = time.perf_counter() - t0
        if dt >= 1.0:
            fps_show = frames / dt
            frames = 0
            t0 = time.perf_counter()

        y = 24
        put_text(frame, f"FPS: {fps_show:.1f} | calib={int(has_calib)} smooth={int(not args.no_smooth)} alpha={alpha:.2f}", y); y += 22
        put_text(frame, f"gaze_raw(y/p)={gaze_yaw}/{gaze_pitch}", y); y += 22
        put_text(frame, f"gaze_rel(y/p)={yaw_rel}/{pitch_rel}", y); y += 22
        put_text(frame, f"gaze_draw(y/p)={yaw_draw}/{pitch_draw}", y); y += 22
        put_text(frame, f"head_yaw_deg={cvf.yaw}", y); y += 22

        cv2.imshow("WebGazeGuard - main", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("c"), ord("C")):
            buf_yaw.clear()
            buf_pitch.clear()
            collecting = True
            collect_until = ts + 1100
            print("[GAZE] Collecting calibration samples (~1s)...")
        if key in (ord("r"), ord("R")):
            has_calib = False
            collecting = False
            ema_y = None
            ema_p = None
            buf_yaw.clear()
            buf_pitch.clear()
            print("[GAZE] Calibration reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
