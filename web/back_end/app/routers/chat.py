from fastapi import APIRouter, Depends
from ..deps import get_current_user_id
from ..schemas import ChatIn, ChatOut
from ..services.chat_adapter import chat_reply

# Pull latest realtime metrics from WebSocket stream history
# (populated by web/back_end/app/routers/analysis.py)
from .analysis import _history as _ws_history  # noqa: E402

router = APIRouter()


@router.post("/chat", response_model=ChatOut)
async def chat(data: ChatIn, user_id: int = Depends(get_current_user_id)):
    # Base context coming from client (optional)
    ctx = data.context or {}

    # Attach latest realtime metrics so the coach can be "metrics-aware"
    try:
        rows = _ws_history.get(int(user_id), [])
        if rows:
            last = rows[-1] or {}
            ctx = dict(ctx)  # copy to avoid mutating input
            ctx["metrics"] = last.get("metrics") or {}
            ctx["ts_ms"] = last.get("ts_ms")
    except Exception:
        # If anything goes wrong, fall back to client-provided context only
        pass

    reply, note = await chat_reply(data.message, context=ctx)
    return ChatOut(reply=reply, safety_note=note)
