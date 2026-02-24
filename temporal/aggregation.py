from __future__ import annotations

from collections import deque
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
        self.buf_ear = deque(maxlen=self.maxlen)
        self.buf_yaw = deque(maxlen=self.maxlen)
        self.buf_pitch = deque(maxlen=self.maxlen)
        self.buf_roll = deque(maxlen=self.maxlen)
        self.buf_dist_cat = deque(maxlen=self.maxlen)

        self._prev_total_blinks = 0
        self.blink_count_window = 0

    def update(self, feat: CVFeatures):
        self.buf_ts.append(int(feat.timestamp_ms))

        if feat.ear_mean is not None:
            self.buf_ear.append(float(feat.ear_mean))
        if feat.yaw is not None:
            self.buf_yaw.append(float(feat.yaw))
        if feat.pitch is not None:
            self.buf_pitch.append(float(feat.pitch))
        if feat.roll is not None:
            self.buf_roll.append(float(feat.roll))
        if feat.distance_cat is not None:
            self.buf_dist_cat.append(str(feat.distance_cat))

        # blink counting via total_blinks delta (stateful)
        tb = int(feat.total_blinks)
        if tb > self._prev_total_blinks:
            self.blink_count_window += (tb - self._prev_total_blinks)
        self._prev_total_blinks = tb

    def ready(self) -> bool:
        return len(self.buf_ts) >= self.maxlen

    def compute(self) -> Optional[WindowFeatures]:
        if not self.buf_ts:
            return None

        def safe_stats(x):
            x = np.asarray(list(x), dtype=np.float32)
            if x.size == 0:
                return {"mean": None, "std": None}
            return {"mean": float(np.mean(x)), "std": float(np.std(x))}

        ear_s = safe_stats(self.buf_ear)
        yaw_s = safe_stats(self.buf_yaw)
        pit_s = safe_stats(self.buf_pitch)
        rol_s = safe_stats(self.buf_roll)

        minutes = self.window_seconds / 60.0
        blink_rate_bpm = (self.blink_count_window / minutes) if minutes > 0 else None

        # mode of distance category
        dist_mode = None
        if len(self.buf_dist_cat) > 0:
            vals = list(self.buf_dist_cat)
            dist_mode = max(set(vals), key=vals.count)

        return WindowFeatures(
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
            blink_rate_bpm=float(blink_rate_bpm) if blink_rate_bpm is not None else None,
            distance_cat_mode=dist_mode,
        )

    def reset_window_counters(self):
        self.blink_count_window = 0
