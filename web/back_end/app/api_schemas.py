from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal

class AuthRegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)

class AuthLoginIn(BaseModel):
    email: EmailStr
    password: str

class AuthOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class MeOut(BaseModel):
    id: int
    email: EmailStr

# Real-time analysis
class Metrics(BaseModel):
    blink_rate_per_min: float
    ear: float
    head_pose_yaw_deg: float
    head_pose_pitch_deg: float
    distance_cm: float
    strain_risk: float  # 0..1
    posture_flag: Optional[Literal["OK", "FORWARD_HEAD", "TILT"]] = None
    gaze_yaw_deg: Optional[float] = None
    gaze_pitch_deg: Optional[float] = None

class FrameAnalysisOut(BaseModel):
    ts_ms: int
    metrics: Metrics

class ChatIn(BaseModel):
    message: str
    context: Optional[dict] = None

class ChatOut(BaseModel):
    reply: str
    safety_note: str | None = None
