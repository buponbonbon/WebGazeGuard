from __future__ import annotations

from collections import deque, Counter
from typing import Optional

import numpy as np

from core.schemas import CVFeatures, WindowFeatures


class WindowAggregator:
    # Rolling window aggregator for CVFeatures -> WindowFeatures

    def __init__(self, fps: float, window_seconds: float = 2.5):
        self.fps = float(fps)
        self.window_seconds = float(window_seconds)
        self.maxlen = int(round(self.fps * self.window_seconds))

        self.buf_ts = deque(maxlen=self.maxlen)

        # keep buffers aligned per-frame (use NaN placeholders)
        self.buf_ear = deque(maxlen=self.maxlen)
        self.buf_yaw = deque(maxlen=self.maxlen)
        self.buf_pitch = deque(maxlen=self.maxlen)
        self.buf_roll = deque(maxlen=self.maxlen)
        self.buf_gaze_yaw = deque(maxlen=self.maxlen)
        self.buf_gaze_pitch = deque(maxlen=self.maxlen)

        # store distance category per-frame (None if missing)
        self.buf_dist_cat = deque(maxlen=self.maxlen)

        # store blink flags per frame instead of cumulative counter
        self.buf_blink_flag = deque(maxlen=self.maxlen)

        # cache last computed window
        self._dirty = True
        self._last_window: Optional[WindowFeatures] = None

    def update(self, feat: CVFeatures):
        self.buf_ts.append(int(feat.timestamp_ms))

        # append per-frame values (NaN when unavailable)
        self.buf_ear.append(float(feat.ear_mean) if feat.ear_mean is not None else float("nan"))
        self.buf_yaw.append(float(feat.yaw) if feat.yaw is not None else float("nan"))
        self.buf_pitch.append(float(feat.pitch) if feat.pitch is not None else float("nan"))
        self.buf_roll.append(float(feat.roll) if feat.roll is not None else float("nan"))
        self.buf_gaze_yaw.append(float(feat.gaze_yaw) if feat.gaze_yaw is not None else float("nan"))
        self.buf_gaze_pitch.append(float(feat.gaze_pitch) if feat.gaze_pitch is not None else float("nan"))

        self.buf_dist_cat.append(str(feat.distance_cat) if feat.distance_cat is not None else None)

        # blink counting per frame (1 if new blink occurred)
        tb = int(feat.total_blinks)
        if not hasattr(self, "_prev_total_blinks"):
            self._prev_total_blinks = tb

        blink_flag = 1 if tb > self._prev_total_blinks else 0
        self.buf_blink_flag.append(blink_flag)
        self._prev_total_blinks = tb

        self._dirty = True

    def ready(self) -> bool:
        return len(self.buf_ts) >= self.maxlen

    def compute(self) -> Optional[WindowFeatures]:
        if not self.buf_ts:
            return None

        # return cached result if no new frames
        if not self._dirty and self._last_window is not None:
            return self._last_window

        def safe_stats(x):
            x = np.asarray(list(x), dtype=np.float32)
            if x.size == 0:
                return {"mean": None, "std": None}
            # ignore NaNs (missing frames)
            if np.all(np.isnan(x)):
                return {"mean": None, "std": None}
            return {"mean": float(np.nanmean(x)), "std": float(np.nanstd(x))}

        ear_s = safe_stats(self.buf_ear)
        yaw_s = safe_stats(self.buf_yaw)
        pit_s = safe_stats(self.buf_pitch)
        rol_s = safe_stats(self.buf_roll)
        gaze_yaw_s = safe_stats(self.buf_gaze_yaw)
        gaze_pitch_s = safe_stats(self.buf_gaze_pitch)

        minutes = self.window_seconds / 60.0

        blink_count_window = int(np.sum(self.buf_blink_flag)) if len(self.buf_blink_flag) > 0 else 0
        blink_rate_bpm = (blink_count_window / minutes) if minutes > 0 else None

        # mode of distance category (ignore None)
        dist_mode = None
        if len(self.buf_dist_cat) > 0:
            vals = [v for v in self.buf_dist_cat if v is not None]
            if vals:
                dist_mode = Counter(vals).most_common(1)[0][0]

        win = WindowFeatures(
            window_start_ms=int(self.buf_ts[0]),
            window_end_ms=int(self.buf_ts[-1]),
            window_seconds=float(self.window_seconds),
            ear_mean=ear_s["mean"],
            ear_std=ear_s["std"],
            yaw_mean=yaw_s["mean"],
            yaw_std=yaw_s["std"],
            pitch_mean=pit_s["mean"],
            pitch_std=pit_s["std"],
            roll_mean=rol_s["mean"],
            roll_std=rol_s["std"],
            gaze_yaw_mean=gaze_yaw_s["mean"],
            gaze_yaw_std=gaze_yaw_s["std"],
            gaze_pitch_mean=gaze_pitch_s["mean"],
            gaze_pitch_std=gaze_pitch_s["std"],
            blink_rate_bpm=float(blink_rate_bpm) if blink_rate_bpm is not None else None,
            distance_cat_mode=dist_mode,
        )

        self._last_window = win
        self._dirty = False
        return win
