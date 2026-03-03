from __future__ import annotations

"""report_summarize.py

Generate short, user-facing coaching messages from a severity / risk level.

This module is intentionally lightweight (no torch/cv2) so it can be imported
from web/core/vision code safely.

Supports:
- int severity IDs (0..3) from NLP (None/Nhẹ/Vừa/Nặng)
- legacy int severity IDs (0..2) from older CV-only pipeline (low/medium/high)
- string labels: "None", "Nhẹ", "Vừa", "Nặng", "low", "medium", "high",
  and RiskOutput.risk_level values like "Low"/"Medium"/"High".
"""

from typing import Dict, Union


# Canonical 4-level scale used by NLP module:
# 0 = none, 1 = mild, 2 = moderate, 3 = severe
SEVERITY_TEXT_4 = {
    0: "none",
    1: "mild",
    2: "moderate",
    3: "severe",
}

# Legacy 3-level scale used by older code:
# 0 = low, 1 = medium, 2 = high
SEVERITY_TEXT_3 = {
    0: "low",
    1: "medium",
    2: "high",
}

# Accept common aliases (Vietnamese + English)
_VI2ID = {"None": 0, "Không": 0, "Nhẹ": 1, "Vừa": 2, "Nặng": 3}
_EN2ID = {
    "none": 0,
    # 3-level labels (legacy / UI)
    "low": 1,       # map low -> mild (since 0 reserved for none)
    "medium": 2,
    "high": 3,
    # 4-level labels (canonical)
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

# Messages for the canonical 4-level scale
MESSAGES_4 = {
    0: {
        "vi": "Hiện chưa ghi nhận dấu hiệu mỏi mắt rõ rệt. Hãy duy trì tư thế đúng và nghỉ mắt định kỳ (quy tắc 20-20-20).",
        "en": "No clear eye strain signs detected. Maintain proper posture and take regular eye breaks (20-20-20 rule).",
    },
    1: {
        "vi": "Bạn có dấu hiệu mỏi mắt nhẹ. Thử chớp mắt chủ động và nghỉ mắt ngắn 20–30 giây.",
        "en": "Mild eye strain signs. Try conscious blinking and a short 20–30s eye break.",
    },
    2: {
        "vi": "Nguy cơ mỏi mắt mức trung bình. Nên nghỉ sau mỗi 20 phút, nhìn xa 20 giây, và điều chỉnh khoảng cách/tư thế.",
        "en": "Moderate eye strain risk. Take breaks every 20 minutes, look far for 20 seconds, and adjust distance/posture.",
    },
    3: {
        "vi": "Mức độ mỏi mắt cao. Nên dừng nhìn màn hình, nghỉ dài hơn, và cân nhắc tham khảo chuyên gia nếu kéo dài.",
        "en": "High eye strain level. Stop screen use, rest longer, and consider professional advice if it persists.",
    },
}

# Optional disclaimer (keep short)
DEFAULT_DISCLAIMER_VI = "Lưu ý: Hệ thống chỉ mang tính hỗ trợ, không thay thế chẩn đoán y khoa."
DEFAULT_DISCLAIMER_EN = "Note: This system is for support only and does not replace medical diagnosis."


SeverityLike = Union[int, str]


def _to_severity_id(severity: SeverityLike) -> int:
    """Normalize severity to canonical ID in [0,3]."""
    if isinstance(severity, int):
        # Heuristic for legacy 0..2 scale: treat it as low/medium/high (no 'none').
        if 0 <= severity <= 2:
            # map 0->1 (mild), 1->2 (moderate), 2->3 (severe)
            return severity + 1
        return max(0, min(3, severity))

    s = str(severity).strip()
    if s in _VI2ID:
        return _VI2ID[s]

    s_lower = s.lower()
    # RiskOutput.risk_level often uses "Low/Medium/High"
    if s_lower in ("low", "medium", "high"):
        return _EN2ID[s_lower]
    if s_lower in _EN2ID:
        return _EN2ID[s_lower]

    # unknown -> moderate as conservative default
    return 2


def summarize_severity(severity: SeverityLike, lang: str = "vi") -> str:
    """Return a short coaching message."""
    sid = _to_severity_id(severity)
    msg_pack = MESSAGES_4.get(sid)
    if not msg_pack:
        return "Không xác định được mức độ mỏi mắt." if lang == "vi" else "Unknown eye strain level."
    return msg_pack.get(lang, msg_pack["vi"])


def build_report(severity: SeverityLike) -> Dict[str, str]:
    """Build a bilingual report payload for UI."""
    sid = _to_severity_id(severity)
    return {
        # Canonical fields
        "severity_id": str(sid),
        "severity_label": SEVERITY_TEXT_4.get(sid, "unknown"),
        "message_vi": summarize_severity(sid, "vi"),
        "message_en": summarize_severity(sid, "en"),
        "disclaimer_vi": DEFAULT_DISCLAIMER_VI,
        "disclaimer_en": DEFAULT_DISCLAIMER_EN,
        # Optional legacy helper (handy if some UI still expects low/medium/high)
        "severity_label_legacy_3": SEVERITY_TEXT_3.get(max(0, sid - 1), "unknown") if sid > 0 else "low",
    }
