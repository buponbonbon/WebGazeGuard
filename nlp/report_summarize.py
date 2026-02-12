from __future__ import annotations
from typing import Dict


SEVERITY_TEXT = {
    0: "low",
    1: "medium",
    2: "high",
}


MESSAGES = {
    0: {
        "vi": "Bạn chỉ có dấu hiệu mỏi mắt nhẹ. Hãy duy trì tư thế đúng và nghỉ mắt định kỳ.",
        "en": "You show mild signs of eye strain. Maintain proper posture and take regular eye breaks.",
    },
    1: {
        "vi": "Nguy cơ mỏi mắt mức trung bình. Nên nghỉ sau mỗi 20 phút và giảm thời gian nhìn màn hình.",
        "en": "Moderate eye strain risk. Take a break every 20 minutes and reduce screen exposure.",
    },
    2: {
        "vi": "Mức độ mỏi mắt cao. Nên dừng sử dụng màn hình và cân nhắc tham khảo ý kiến chuyên gia.",
        "en": "High eye strain level. Stop screen usage and consider consulting a specialist if symptoms persist.",
    },
}


def summarize_severity(severity: int, lang: str = "vi") -> str:
    if severity not in MESSAGES:
        return "Không xác định được mức độ mỏi mắt." if lang == "vi" else "Unknown eye strain level."
    return MESSAGES[severity].get(lang, MESSAGES[severity]["vi"])


def build_report(severity: int) -> Dict[str, str]:
    return {
        "severity_label": SEVERITY_TEXT.get(severity, "unknown"),
        "message_vi": summarize_severity(severity, "vi"),
        "message_en": summarize_severity(severity, "en"),
    }
