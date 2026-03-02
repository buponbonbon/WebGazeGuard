import time
import jwt
from jwt import PyJWTError

from ..settings import settings

def create_token(user_id: int) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + settings.jwt_expires_min * 60,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")

def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except PyJWTError:
        return None
