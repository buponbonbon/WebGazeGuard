# ===== Eye landmark subset and EAR computation =====
# This module extracts periocular landmarks and computes the Eye Aspect Ratio (EAR),
# which serves as a geometry-based indicator for blink detection and eye closure analysis.

import numpy as np

# MediaPipe FaceMesh / FaceLandmarker indices for eye contours (standard convention)
# Left eye (subject's left): 6-point EAR subset
LEFT_EYE_EAR_IDX  = [33, 160, 158, 133, 153, 144]   # [p1, p2, p3, p4, p5, p6]
# Right eye (subject's right): 6-point EAR subset
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]  # [p1, p2, p3, p4, p5, p6]

def _euclid(p, q):
    """Compute Euclidean distance between two 2D points (x, y)."""
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    return float(np.linalg.norm(p - q))

def compute_ear(lms_xy, eye_idx):
    """
    Compute Eye Aspect Ratio (EAR) from 6 eye landmarks.

    Parameters
    ----------
    lms_xy : dict
        Dictionary mapping landmark indices to (x, y) pixel coordinates.
    eye_idx : list[int]
        Six indices defining the eye contour points: [p1, p2, p3, p4, p5, p6].

    Returns
    -------
    float
        Eye Aspect Ratio (EAR).
    """
    p1, p2, p3, p4, p5, p6 = [lms_xy[i] for i in eye_idx]
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    num = _euclid(p2, p6) + _euclid(p3, p5)
    den = 2.0 * _euclid(p1, p4) + 1e-6
    return num / den

def compute_ear_both_eyes(lms_xy):
    """
    Compute EAR for left eye, right eye, and their average.

    Returns
    -------
    (ear_left, ear_right, ear_mean) : tuple(float, float, float)
    """
    ear_l = compute_ear(lms_xy, LEFT_EYE_EAR_IDX)
    ear_r = compute_ear(lms_xy, RIGHT_EYE_EAR_IDX)
    return ear_l, ear_r, 0.5 * (ear_l + ear_r)

print("EAR computation module initialized.")
