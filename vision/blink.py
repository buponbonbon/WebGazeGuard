
# This module converts frame-level EAR measurements into blink events using thresholding with temporal persistence to reduce spurious detections.

import time

EAR_THRESH = 0.20          # initial default; should be calibrated per camera/user
EAR_CONSEC_FRAMES = 2      # minimum consecutive frames below threshold to count a blink

# Hysteresis factors (relative to baseline)
EAR_CLOSE_RATIO = 0.75     # eye considered closed if EAR < baseline * this
EAR_OPEN_RATIO = 0.85      # eye considered open if EAR > baseline * this

REFRACTORY_MS = 250        # minimum time between two blink events


class BlinkDetector:

    #Blink detector based on EAR thresholding with hysteresis and refractory period.

    def __init__(self, ear_thresh=EAR_THRESH, consec_frames=EAR_CONSEC_FRAMES):
        self.ear_thresh = float(ear_thresh)
        self.consec_frames = int(consec_frames)

        self._state = "OPEN"          # OPEN or CLOSED
        self._below_count = 0
        self._last_blink_ts = 0

        self._baseline_samples = []
        self._ear_baseline = None

        self.total_blinks = 0

    def _update_baseline(self, ear_mean):
        # Collect initial baseline for open-eye EAR
        if self._ear_baseline is not None:
            return

        self._baseline_samples.append(ear_mean)

        if len(self._baseline_samples) >= 60:  # ~2s at 30 FPS
            sorted_vals = sorted(self._baseline_samples)
            mid = len(sorted_vals) // 2
            self._ear_baseline = sorted_vals[mid]

    def update(self, ear_mean, timestamp_ms=None):
        blink = False
        now_ms = int(timestamp_ms) if timestamp_ms is not None else int(time.time() * 1000)

        if ear_mean is None:
            return False

        # Baseline calibration phase
        if self._ear_baseline is None:
            self._update_baseline(ear_mean)

        baseline = self._ear_baseline if self._ear_baseline else ear_mean

        close_thresh = baseline * EAR_CLOSE_RATIO
        open_thresh = baseline * EAR_OPEN_RATIO

        # CLOSED detection
        if ear_mean < close_thresh:
            self._below_count += 1

            if self._state == "OPEN" and self._below_count >= self.consec_frames:
                self._state = "CLOSED"

        else:
            # Eye reopening
            if self._state == "CLOSED" and ear_mean > open_thresh:
                # Refractory check
                if now_ms - self._last_blink_ts > REFRACTORY_MS:
                    self.total_blinks += 1
                    blink = True
                    self._last_blink_ts = now_ms

                self._state = "OPEN"

            self._below_count = 0

        return blink


blink_detector = BlinkDetector()
print("Blink detector initialized.")