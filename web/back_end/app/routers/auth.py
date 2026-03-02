from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..deps import get_db
from ..schemas import AuthRegisterIn, AuthLoginIn, AuthOut
from ..models.user import User
from ..utils.password import hash_password, verify_password
from ..utils.jwt import create_token

router = APIRouter()

@router.post("/register", response_model=AuthOut)
def register(data: AuthRegisterIn, db: Session = Depends(get_db)):
    exists = db.scalar(select(User).where(User.email == data.email))
    if exists:
        raise HTTPException(status_code=409, detail="Email already registered")
    u = User(email=data.email, password_hash=hash_password(data.password))
    db.add(u)
    db.commit()
    db.refresh(u)
    return AuthOut(access_token=create_token(u.id))

@router.post("/login", response_model=AuthOut)
def login(data: AuthLoginIn, db: Session = Depends(get_db)):
    u = db.scalar(select(User).where(User.email == data.email))
    if not u or not verify_password(data.password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return AuthOut(access_token=create_token(u.id))
