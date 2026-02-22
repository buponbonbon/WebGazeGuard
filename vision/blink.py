# ===== Blink event detection (EAR thresholding) =====
# This module converts frame-level EAR measurements into blink events using
# thresholding with temporal persistence to reduce spurious detections.

EAR_THRESH = 0.20          # initial default; should be calibrated per camera/user
EAR_CONSEC_FRAMES = 2      # minimum consecutive frames below threshold to count a blink

class BlinkDetector:
    """
    Blink detector based on EAR thresholding with a consecutive-frame constraint.
    """
    def __init__(self, ear_thresh=EAR_THRESH, consec_frames=EAR_CONSEC_FRAMES):
        self.ear_thresh = float(ear_thresh)
        self.consec_frames = int(consec_frames)
        self._below_count = 0
        self.total_blinks = 0

    def update(self, ear_mean):
        """
        Update detector with current EAR value.

        Parameters
        ----------
        ear_mean : float
            Mean EAR value for the current frame.

        Returns
        -------
        blink : bool
            True if a blink event is registered at this update step.
        """
        blink = False
        if ear_mean < self.ear_thresh:
            self._below_count += 1
        else:
            if self._below_count >= self.consec_frames:
                self.total_blinks += 1
                blink = True
            self._below_count = 0
        return blink

blink_detector = BlinkDetector()
print("Blink detector initialized.")
