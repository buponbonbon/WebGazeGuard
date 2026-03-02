"""Risk engine: fuse features into a simple strain risk score (0..1).

From the PDF idea (page 1-2), the system estimates strain using:
- Blink rate
- Head pose
- Distance to screen
- (Optional) gaze direction + user text

Here we implement a lightweight, interpretable rule-based risk score so it runs fast.
You can replace this with your own model (MLP/XGBoost) later.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RiskOut:
    strain_risk: float
    posture_flag: str

def compute_risk(blink_rate_per_min: float, yaw_deg: float, pitch_deg: float, distance_cm: float) -> RiskOut:
    # Blink: low blink increases risk
    blink_risk = 0.0
    if blink_rate_per_min < 8:
        blink_risk = 0.7
    elif blink_rate_per_min < 12:
        blink_risk = 0.4
    else:
        blink_risk = 0.15

    # Head pose: large pitch/yaw increases risk
    pose_mag = (abs(yaw_deg) / 25.0 + abs(pitch_deg) / 20.0) / 2.0
    pose_risk = min(1.0, pose_mag)

    # Distance: too close increases risk
    if distance_cm < 45:
        dist_risk = 0.8
    elif distance_cm < 55:
        dist_risk = 0.45
    else:
        dist_risk = 0.2

    # Weighted fuse
    risk = 0.5*blink_risk + 0.3*pose_risk + 0.2*dist_risk
    risk = max(0.0, min(1.0, risk))

    posture = "OK"
    if distance_cm < 50:
        posture = "FORWARD_HEAD"
    if abs(yaw_deg) > 12 or abs(pitch_deg) > 10:
        posture = "TILT"

    return RiskOut(strain_risk=risk, posture_flag=posture)
