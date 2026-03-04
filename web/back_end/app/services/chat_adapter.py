
"""Chat adapter with built‑in PhoBERT checkpoint path."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import httpx
from ..settings import settings
from nlp.report_summarize import summarize_severity, DEFAULT_DISCLAIMER_VI

# ----------- CHANGE: default checkpoint path -----------
DEFAULT_PHOBERT_PATH = r"C:\Users\ASUS\Desktop\WebGazeGuard\nlp\checkpoints\best_model_PhoBert.pt"
# -------------------------------------------------------

_NLP_MODEL = None
_NLP_MODEL_PATH = None


def _get_local_model_path() -> Optional[str]:
    """
    Determine PhoBERT checkpoint path.
    Priority:
    1. Environment variable NLP_MODEL_PATH
    2. settings.* field
    3. DEFAULT_PHOBERT_PATH (hardcoded)
    """

    env = os.getenv("NLP_MODEL_PATH")
    if env:
        return env

    for name in (
        "nlp_model_path",
        "phobert_model_path",
        "phobert_ckpt_path",
        "nlp_ckpt_path",
        "nlp_checkpoint",
    ):
        try:
            val = getattr(settings, name)
            if isinstance(val, str) and val.strip():
                return val.strip()
        except Exception:
            pass

    # fallback to hardcoded path
    return DEFAULT_PHOBERT_PATH


def _ensure_local_nlp_model():
    global _NLP_MODEL, _NLP_MODEL_PATH

    model_path = _get_local_model_path()

    if not model_path:
        return None

    if _NLP_MODEL is not None and _NLP_MODEL_PATH == model_path:
        return _NLP_MODEL

    try:
        from nlp.classifier import get_classifier

        _NLP_MODEL = get_classifier(model_path, device="cpu")
        _NLP_MODEL_PATH = model_path
        print(f"[NLP] PhoBERT loaded from: {model_path}")
        return _NLP_MODEL
    except Exception as e:
        print("[NLP] Failed to load PhoBERT:", e)
        _NLP_MODEL = None
        return None


def _coaching_from_metrics(context: Dict[str, Any]) -> str:
    m = context.get("metrics") or {}
    lines = []

    br = m.get("blink_rate_per_min")
    dist = m.get("distance_cm")
    yaw = m.get("head_pose_yaw_deg")
    pitch = m.get("head_pose_pitch_deg")
    risk = m.get("strain_risk")

    if br and br < 8:
        lines.append(f"- Blink rate thấp (~{br:.0f}/phút). Thử chớp mắt chủ động.")
    if dist and dist < 45:
        lines.append(f"- Bạn đang ngồi khá gần (~{dist:.0f}cm). Nên lùi ra 50–70cm.")
    if yaw and abs(yaw) > 15:
        lines.append("- Tư thế đầu hơi lệch. Thử giữ cổ thẳng.")
    if risk and risk > 0.6:
        lines.append("- Nguy cơ mỏi mắt cao. Nghỉ mắt 20–30 giây.")

    return "\n".join(lines)


async def chat_reply(message: str, context: dict | None = None) -> Tuple[str, str | None]:

    if getattr(settings, "ai_nlp_endpoint", None):
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.post(
                str(settings.ai_nlp_endpoint),
                json={"message": message, "context": context or {}},
            )
            r.raise_for_status()
            data = r.json()
            return str(data.get("reply", "")), data.get("safety_note")

    ctx: Dict[str, Any] = context or {}

    model = _ensure_local_nlp_model()

    if model is not None:
        try:
            feats = model.predict_features(message, device="cpu")

            sev = getattr(feats, "severity_label", "Vừa")
            conf = float(getattr(feats, "confidence", 0.0))

            base_msg = summarize_severity(sev, lang="vi")
            extra = _coaching_from_metrics(ctx)

            reply = (
                f"Kết quả NLP: {sev} (confidence ~{conf*100:.0f}%).\n"
                f"{base_msg}\n\n{extra}"
            )

            return reply, DEFAULT_DISCLAIMER_VI

        except Exception as e:
            print("[NLP] inference error:", e)

    return (
        "Bạn mô tả thêm cảm giác của mắt (khô, rát, nhức…) để mình hỗ trợ tốt hơn nhé.",
        None,
    )
