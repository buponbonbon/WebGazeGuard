from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..deps import get_db, get_current_user_id
from ..models.user import User
from ..schemas import MeOut

router = APIRouter()

@router.get("", response_model=MeOut)
def me(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    u = db.scalar(select(User).where(User.id == user_id))
    return MeOut(id=u.id, email=u.email)
