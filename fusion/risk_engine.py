from __future__ import annotations
from core.schemas import CVFeatures, WindowFeatures, NLPFeatures, RiskOutput
from typing import Optional, Dict, Tuple, List
import re

from unicodedata import normalize as _ud_normalize


# scoring helpers

# --- NLP guardrail helpers (lightweight, no extra deps) ---
_SEVERE_KWS = [
    "đau rát", "đau mắt", "mờ", "mờ mắt", "nhức đầu", "đỏ", "mắt đỏ",
    "chảy nước mắt", "cộm", "xốn", "rát", "căng đau",
]


def _strip_accents(s: str) -> str:
    s = _ud_normalize('NFKD', s)
    return ''.join(ch for ch in s if not getattr(ch, 'combining', lambda: False)())


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _has_severe_keyword(text: str) -> bool:
    t = _norm_text(text)
    t2 = _strip_accents(t)
    for kw in _SEVERE_KWS:
        k = _norm_text(kw)
        if k in t or _strip_accents(k) in t2:
            return True
    return False



def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _lin_score(x: float, lo: float, hi: float) -> float:
    # linear ramp
    if hi <= lo:
        return 0.0
    return _clip01((x - lo) / (hi - lo))


def _dist_risk(distance_cat_mode: Optional[str]) -> Tuple[float, Optional[str]]:
    # category from vision.head_distance: "too_close" | "normal" | "too_far"
    if not distance_cat_mode:
        return 0.0, None
    cat = distance_cat_mode.lower()
    if cat == "too_close":
        return 1.0, "too_close"
    if cat == "too_far":
        return 0.4, "too_far"
    return 0.0, None


def _blink_risk(blink_rate_bpm: Optional[float]) -> float:
    # heuristic: very low blink rate correlates with eye strain/dryness
    # typical relaxed blink rate ~ 10-20 bpm; below ~7 bpm is concerning
    if blink_rate_bpm is None:
        return 0.0
    br = float(blink_rate_bpm)
    # low blink -> higher risk
    low = 7.0
    ok = 15.0
    return _clip01(1.0 - _lin_score(br, low, ok))


def _posture_risk(pitch_mean: Optional[float], yaw_mean: Optional[float]) -> Tuple[float, Optional[str]]:
    # pitch/yaw are degrees in your pipeline
    # large absolute angles suggest poor posture/neck strain
    # map flags to web Metrics schema: OK | FORWARD_HEAD | TILT
    if pitch_mean is None and yaw_mean is None:
        return 0.0, None

    p = abs(float(pitch_mean)) if pitch_mean is not None else 0.0
    y = abs(float(yaw_mean)) if yaw_mean is not None else 0.0

    # conservative thresholds
    pitch_s = _lin_score(p, lo=10.0, hi=25.0)
    yaw_s = _lin_score(y, lo=15.0, hi=35.0)

    score = _clip01(0.6 * pitch_s + 0.4 * yaw_s)

    if score < 0.7:
        return score, None
    if p >= y:
        return score, "FORWARD_HEAD"
    return score, "TILT"




def _gaze_risk(cv: Optional[CVFeatures]) -> Tuple[float, Optional[str]]:
    """
    Lightweight gaze contribution from the latest frame only.
    Keep this intentionally weak so it supplements, not overrides,
    blink/distance/posture.
    """
    if cv is None:
        return 0.0, None

    gaze_yaw = getattr(cv, "gaze_yaw", None)
    gaze_pitch = getattr(cv, "gaze_pitch", None)
    if gaze_yaw is None and gaze_pitch is None:
        return 0.0, None

    gy = abs(float(gaze_yaw)) if gaze_yaw is not None else 0.0
    gp = abs(float(gaze_pitch)) if gaze_pitch is not None else 0.0

    # Conservative thresholds to avoid overreacting to noisy gaze estimates.
    yaw_s = _lin_score(gy, lo=12.0, hi=25.0)
    pitch_s = _lin_score(gp, lo=8.0, hi=18.0)
    score = _clip01(0.65 * yaw_s + 0.35 * pitch_s)

    if score < 0.6:
        return score, None
    return score, "gaze_off_center"


def _nlp_risk(nlp: Optional[NLPFeatures]) -> Tuple[float, Optional[str], float]:
    # map discomfort_level
    if nlp is None:
        return 0.0, None, 0.0

    lvl = int(getattr(nlp, "discomfort_level", 0) or 0)
    conf = float(getattr(nlp, "confidence", 0.0) or 0.0)
    text = str(getattr(nlp, "original_text", "") or "")

    # --- Guardrail: prevent underestimation for severe symptom keywords ---
    # This is intentionally lightweight to keep realtime stable.
    if lvl < 2 and text and _has_severe_keyword(text):
        lvl = 2

    # normalize levels:

    if lvl <= 0:
        base = 0.0
        lab = "None"
    elif lvl == 1:
        base = 0.25
        lab = "Nhẹ"
    elif lvl == 2:
        base = 0.55
        lab = "Vừa"
    else:
        base = 0.85
        lab = "Nặng"

    # confidence gates impact
    score = _clip01(base * (0.5 + 0.5 * conf))  # conf in [0,1] -> scale [0.5,1.0]
    return score, lab, conf


def _combine_scores(components: Dict[str, float], weights: Dict[str, float]) -> float:
    s = 0.0
    wsum = 0.0
    for k, v in components.items():
        w = float(weights.get(k, 0.0))
        s += w * float(v)
        wsum += w
    return float(s / wsum) if wsum > 0 else 0.0


def _level_from_score(score: float) -> str:
    # thresholds tuned to be conservative
    if score >= 0.66:
        return "High"
    if score >= 0.33:
        return "Medium"
    return "Low"


# public API

def assess_risk(
    window: WindowFeatures,
    nlp: Optional[NLPFeatures] = None,
    *,
    weights: Optional[Dict[str, float]] = None,
    cv: Optional[CVFeatures] = None,
) -> RiskOutput:

    # default weights
    w = weights or {
        "blink": 0.25,
        "distance": 0.25,
        "posture": 0.20,
        "gaze": 0.10,
        "nlp": 0.20,
    }

    blink_s = _blink_risk(window.blink_rate_bpm)
    dist_s, dist_flag = _dist_risk(window.distance_cat_mode)
    post_s, posture_flag = _posture_risk(window.pitch_mean, window.yaw_mean)
    gaze_s, gaze_flag = _gaze_risk(cv)
    nlp_s, nlp_lab, _ = _nlp_risk(nlp)

    components = {
        "blink": blink_s,
        "distance": dist_s,
        "posture": post_s,
        "gaze": gaze_s,
        "nlp": nlp_s,
    }

    score = _combine_scores(components, w)
    level = _level_from_score(score)

    # compact explanation string for UI/logging
    exp_bits: List[str] = []
    if blink_s >= 0.6:
        exp_bits.append("blink_rate_low")
    if dist_flag:
        exp_bits.append(dist_flag)
    if posture_flag in ("FORWARD_HEAD", "TILT"):
        exp_bits.append(posture_flag)
    if gaze_flag:
        exp_bits.append(gaze_flag)
    if nlp_lab and nlp_lab != "None" and nlp_s >= 0.4:
        exp_bits.append(f"symptom_{nlp_lab}")

    explanation = ", ".join(exp_bits) if exp_bits else None

    # short recommendation
    rec = None
    if level == "High":
        rec = "Nghỉ mắt 5–10 phút, nhìn xa 20 feet trong 20 giây, điều chỉnh tư thế và khoảng cách màn hình."
    elif level == "Medium":
        rec = "Nghỉ mắt ngắn 20–30 giây, chớp mắt chủ động, giữ khoảng cách và tư thế ổn định."

    disclaimer = None
    if level == "High":
        disclaimer = "Hệ thống chỉ hỗ trợ cảnh báo mỏi mắt, không thay thế chẩn đoán y tế. Nếu triệu chứng kéo dài, hãy gặp bác sĩ."

    return RiskOutput(
        risk_level=level,
        risk_score=float(score),
        risk_components={k: float(v) for k, v in components.items()},
        posture_flag=posture_flag,
        explanation=explanation,
        recommendation=rec,
        disclaimer=disclaimer,
    )
