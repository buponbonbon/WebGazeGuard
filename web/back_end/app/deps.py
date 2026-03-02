from typing import Generator
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from .db import SessionLocal
from .utils.jwt import decode_token

security = HTTPBearer(auto_error=False)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user_id(
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> int:
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    payload = decode_token(creds.credentials)
    if payload is None or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return int(payload["sub"])
