"""Chat adapter.

- If AI_NLP_ENDPOINT is configured, we forward messages to your chatbot service.
- Otherwise, we provide a safe, non-diagnostic coaching template based on the PDF notes
  (20-20-20, blink intentionally, adjust distance/posture).
"""
from __future__ import annotations
import httpx
from ..settings import settings

async def chat_reply(message: str, context: dict | None = None) -> tuple[str, str | None]:
    if settings.ai_nlp_endpoint:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.post(settings.ai_nlp_endpoint, json={"message": message, "context": context or {}})
            r.raise_for_status()
            data = r.json()
            return str(data.get("reply", "")), data.get("safety_note")

    msg = message.lower()
    if "khô" in msg or "dry" in msg:
        return ("Bạn có thể thử quy tắc 20-20-20 (mỗi 20 phút nhìn xa ~6m trong 20 giây), "
                "chớp mắt chủ động vài lần, và cân nhắc tăng độ ẩm phòng/giảm gió quạt thổi trực tiếp. "
                "Nếu khó chịu kéo dài, bạn nên gặp chuyên gia mắt để được tư vấn phù hợp."),                "Lưu ý: Đây là gợi ý thói quen, không phải chẩn đoán y khoa."
    if "đau" in msg or "nhức" in msg or "headache" in msg:
        return ("Nếu bạn nhức đầu kèm mỏi mắt, hãy nghỉ ngắn 1–2 phút, "
                "giảm độ sáng màn hình, giữ khoảng cách ~50–70cm, và kiểm tra tư thế cổ/ vai. "
                "Bạn cũng có thể dùng nước uống và chớp mắt nhiều hơn."),                "Nếu có dấu hiệu nặng/đột ngột, hãy đi khám."
    return ("Mình có thể hỗ trợ bạn theo 3 hướng: (1) thói quen 20-20-20, (2) tư thế và khoảng cách, "
            "(3) nhắc nhở chớp mắt/giảm khô mắt. Bạn mô tả thêm cảm giác hiện tại (mỏi, khô, nhức…) nhé."), None
