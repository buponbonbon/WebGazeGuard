import time
import jwt
from jwt import PyJWTError

from ..settings import settings

ONE_YEAR_SECONDS = 365 * 24 * 60 * 60

def create_token(user_id: int) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + ONE_YEAR_SECONDS,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except PyJWTError:
        return None