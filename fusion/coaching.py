from __future__ import annotations

from typing import Optional, List

from core.schemas import RiskOutput


def _bullets(items: List[str]) -> str:
    return "\n".join([f"- {x}" for x in items if x])


def generate_coaching(risk: RiskOutput) -> str:
    # keep response short, actionable, non-medical
    level = (risk.risk_level or "").strip()
    expl = (risk.explanation or "").strip()

    tips: List[str] = []

    # generic 20-20-20 rule
    tips.append("Áp dụng quy tắc 20-20-20: mỗi 20 phút nhìn xa 20 feet trong 20 giây.")
    tips.append("Chớp mắt chủ động và nghỉ ngắn 20–30 giây khi cảm thấy khô/rát.")

    if "too_close" in expl:
        tips.append("Tăng khoảng cách mắt–màn hình (thường ~50–70 cm) và nâng màn hình ngang tầm mắt.")
    if "too_far" in expl:
        tips.append("Đưa màn hình về khoảng cách thoải mái (không quá xa) để tránh nheo mắt.")
    if "poor_posture" in expl:
        tips.append("Giữ lưng thẳng, vai thả lỏng; điều chỉnh ghế/bàn để cổ không cúi/ngửa lâu.")
    if "blink_rate_low" in expl:
        tips.append("Nếu bạn ít chớp mắt khi tập trung, đặt nhắc nhở chớp mắt 1–2 phút/lần.")
    if "symptom_Nặng" in expl or level == "High":
        tips.append("Nếu triệu chứng kéo dài hoặc đau/nhìn mờ tăng, hãy nghỉ dài hơn và cân nhắc kiểm tra mắt.")

    if level == "High":
        header = "⚠️ Nguy cơ mỏi mắt: CAO"
    elif level == "Medium":
        header = "Nguy cơ mỏi mắt: VỪA"
    else:
        header = "Nguy cơ mỏi mắt: THẤP"

    msg = header + "\n" + _bullets(tips[:5])

    if risk.disclaimer:
        msg += "\n\n" + risk.disclaimer

    return msg
