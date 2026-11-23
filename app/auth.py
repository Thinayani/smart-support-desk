"""Minimal login: one team account from env, HMAC-signed expiring bearer tokens (stdlib only)."""
import base64
import hashlib
import hmac
import time
from typing import Optional

from fastapi import Depends, Header, HTTPException

from . import config


def _sign(payload: str) -> str:
    return hmac.new(config.SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()


def check_credentials(username: str, password: str) -> bool:
    # compare both fields in constant time, and always evaluate both
    user_ok = hmac.compare_digest(username.encode(), config.ADMIN_USER.encode())
    pass_ok = hmac.compare_digest(password.encode(), config.ADMIN_PASSWORD.encode())
    return user_ok and pass_ok


def make_token(username: str, now: Optional[float] = None) -> str:
    exp = int((now or time.time()) + config.TOKEN_TTL_SECONDS)
    payload = f"{username}|{exp}"
    raw = f"{payload}|{_sign(payload)}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def verify_token(token: str, now: Optional[float] = None) -> Optional[str]:
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        username, exp, sig = raw.rsplit("|", 2)
        if not hmac.compare_digest(sig, _sign(f"{username}|{exp}")):
            return None
        if int(exp) < (now or time.time()):
            return None
        return username
    except Exception:
        return None


def require_team(authorization: Optional[str] = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Login required.")
    user = verify_token(authorization[7:].strip())
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid. Please log in again.")
    return user


TeamUser = Depends(require_team)
