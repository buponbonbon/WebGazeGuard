from fastapi import APIRouter, Depends
from ..deps import get_current_user_id
from ..schemas import ChatIn, ChatOut
from ..services.chat_adapter import chat_reply

router = APIRouter()

@router.post("/chat", response_model=ChatOut)
async def chat(data: ChatIn, user_id: int = Depends(get_current_user_id)):
    reply, note = await chat_reply(data.message, context=data.context)
    return ChatOut(reply=reply, safety_note=note)
