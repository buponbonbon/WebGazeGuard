from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
import cv2

from core.config import Config
from core.pipeline import build_pipeline, step


def now_ms() -> int:
    return int(time.time() * 1000)


def put_text(img, text: str, y: int) -> None:
    cv2.putText(
        img,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--fps", type=float, default=30.0, help="Assumed FPS for temporal window")
    ap.add_argument("--window", type=float, default=2.5, help="Rolling window seconds (methodology: 2–3s)")
    ap.add_argument("--ear-thresh", type=float, default=0.20, help="EAR threshold for blink")
    ap.add_argument("--ear-consec", type=int, default=2, help="Consecutive frames under EAR threshold to count blink")
    ap.add_argument("--z0", type=float, default=60.0, help="Distance calibration Z0 (cm)")
    ap.add_argument("--s0", type=float, default=120.0, help="Distance calibration s0 (px)")
    ap.add_argument("--log", type=str, default="", help="CSV log path (optional). Use 'auto' to auto-name.")
    args = ap.parse_args()

    cfg = Config(
        fps_assumed=args.fps,
        window_seconds=args.window,
        ear_thresh=args.ear_thresh,
        ear_consec_frames=args.ear_consec,
        distance_Z0_cm=args.z0,
        distance_s0_px=args.s0,
    )

    state = build_pipeline(cfg)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam}")

    # Optional CSV logging
    log_enabled = False
    log_fp = None
    log_writer = None

    def open_logger() -> None:
        nonlocal log_enabled, log_fp, log_writer
        if log_enabled:
            return
        log_path = args.log
        if log_path.lower() == "auto":
            Path("logs").mkdir(parents=True, exist_ok=True)
            log_path = os.path.join("logs", f"realtime_{int(time.time())}.csv")
        if not log_path:
            return  # user didn't request logging

        log_fp = open(log_path, "w", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(
            log_fp,
            fieldnames=[
                # frame-level
                "timestamp_ms",
                "face_detected",
                "ear_left",
                "ear_right",
                "ear_mean",
                "blink",
                "total_blinks",
                "yaw",
                "pitch",
                "roll",
                "distance_cm",
                "distance_cat",
                # window-level (optional)
                "win_window_start_ms",
                "win_window_end_ms",
                "win_window_seconds",
                "win_ear_mean",
                "win_ear_std",
                "win_yaw_mean",
                "win_yaw_std",
                "win_pitch_mean",
                "win_pitch_std",
                "win_roll_mean",
                "win_roll_std",
                "win_blink_rate_bpm",
                "win_distance_cat_mode",
            ],
        )
        log_writer.writeheader()
        log_enabled = True
        print(f"[LOG] Writing to: {log_path}")

    def close_logger() -> None:
        nonlocal log_enabled, log_fp, log_writer
        if log_fp:
            log_fp.close()
        log_fp = None
        log_writer = None
        log_enabled = False
        print("[LOG] Off")

    t0 = time.perf_counter()
    frames = 0
    fps_show = 0.0

    print("Controls: [Q] quit | [S] toggle CSV log (requires --log path or --log auto)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts = now_ms()
        cvf, win = step(state, frame, timestamp_ms=ts)

        # FPS estimate
        frames += 1
        dt = time.perf_counter() - t0
        if dt >= 1.0:
            fps_show = frames / dt
            frames = 0
            t0 = time.perf_counter()

        # Overlay
        y = 24
        put_text(frame, f"FPS: {fps_show:.1f} | window={cfg.window_seconds:.1f}s", y); y += 22
        put_text(frame, f"face={cvf.face_detected}  ear={cvf.ear_mean if cvf.ear_mean is not None else 'NA'}", y); y += 22
        put_text(frame, f"blink={cvf.blink} total={cvf.total_blinks} dist={cvf.distance_cm if cvf.distance_cm is not None else 'NA'} ({cvf.distance_cat})", y); y += 22
        put_text(frame, f"yaw/pitch/roll = {cvf.yaw if cvf.yaw is not None else 'NA'} / {cvf.pitch if cvf.pitch is not None else 'NA'} / {cvf.roll if cvf.roll is not None else 'NA'}", y); y += 22

        if win is not None:
            put_text(frame, f"[WIN] blink_rate_bpm={win.blink_rate_bpm if win.blink_rate_bpm is not None else 'NA'}  dist_mode={win.distance_cat_mode}", y); y += 22

        # Logging
        if log_enabled and log_writer is not None:
            row = {
                "timestamp_ms": cvf.timestamp_ms,
                "face_detected": cvf.face_detected,
                "ear_left": cvf.ear_left,
                "ear_right": cvf.ear_right,
                "ear_mean": cvf.ear_mean,
                "blink": cvf.blink,
                "total_blinks": cvf.total_blinks,
                "yaw": cvf.yaw,
                "pitch": cvf.pitch,
                "roll": cvf.roll,
                "distance_cm": cvf.distance_cm,
                "distance_cat": cvf.distance_cat,
            }
            if win is not None:
                row.update(
                    {
                        "win_window_start_ms": win.window_start_ms,
                        "win_window_end_ms": win.window_end_ms,
                        "win_window_seconds": win.window_seconds,
                        "win_ear_mean": win.ear_mean,
                        "win_ear_std": win.ear_std,
                        "win_yaw_mean": win.yaw_mean,
                        "win_yaw_std": win.yaw_std,
                        "win_pitch_mean": win.pitch_mean,
                        "win_pitch_std": win.pitch_std,
                        "win_roll_mean": win.roll_mean,
                        "win_roll_std": win.roll_std,
                        "win_blink_rate_bpm": win.blink_rate_bpm,
                        "win_distance_cat_mode": win.distance_cat_mode,
                    }
                )
            else:
                # keep columns present
                row.update({k: None for k in [
                    "win_window_start_ms","win_window_end_ms","win_window_seconds",
                    "win_ear_mean","win_ear_std","win_yaw_mean","win_yaw_std",
                    "win_pitch_mean","win_pitch_std","win_roll_mean","win_roll_std",
                    "win_blink_rate_bpm","win_distance_cat_mode"
                ]})
            log_writer.writerow(row)

        cv2.imshow("WebGazeGuard - main", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("s"), ord("S")):
            if not log_enabled:
                open_logger()
            else:
                close_logger()

    close_logger()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()