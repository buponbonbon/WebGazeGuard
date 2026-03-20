from typing import Generator
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from .db import SessionLocal

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
    # DEMO MODE: bypass auth completely
    return 1