"""JWT helpers + Bearer auth dependency for /api/mobile/v1.

M1 auth: shared password → pair of JWTs (access 15 min, refresh 30 d).
No user accounts; tokens carry only `typ`, `iat`, `exp`, `jti`.

Also provides a short-lived HMAC URL signer for report downloads so
native viewers can open the PDF/DOCX without a Bearer header.
"""
from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from typing import Literal

import jwt
from fastapi import Depends, HTTPException, Request, status

from app.config import get_settings


ACCESS_TTL_SECONDS = 15 * 60
REFRESH_TTL_SECONDS = 30 * 24 * 3600
JWT_ALGORITHM = "HS256"
SIGNED_URL_TTL_SECONDS = 300


def _secret() -> str:
    return get_settings().session_secret


def _now() -> int:
    return int(time.time())


def _encode(typ: Literal["access", "refresh"], ttl: int) -> str:
    now = _now()
    payload = {
        "typ": typ,
        "iat": now,
        "exp": now + ttl,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, _secret(), algorithm=JWT_ALGORITHM)


def encode_access_token() -> str:
    return _encode("access", ACCESS_TTL_SECONDS)


def encode_refresh_token() -> str:
    return _encode("refresh", REFRESH_TTL_SECONDS)


def decode_token(token: str, expected_typ: Literal["access", "refresh"]) -> dict:
    try:
        payload = jwt.decode(token, _secret(), algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    if payload.get("typ") != expected_typ:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Wrong token type")
    return payload


def require_mobile_user(request: Request) -> dict:
    """FastAPI dependency — extracts + validates the Bearer access token."""
    header = request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(token, expected_typ="access")


# ---- Signed download URLs -------------------------------------------------

def _sign_key() -> bytes:
    # Derive a distinct key so an access JWT signature can never collide
    # with a URL signature.
    return hashlib.sha256(b"mobile-url|" + _secret().encode()).digest()


def sign_path(path: str, ttl: int = SIGNED_URL_TTL_SECONDS) -> str:
    exp = _now() + ttl
    msg = f"{path}|{exp}".encode()
    sig = hmac.new(_sign_key(), msg, hashlib.sha256).hexdigest()[:32]
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}exp={exp}&sig={sig}"


def verify_signed_path(path: str, exp: int, sig: str) -> bool:
    if _now() > exp:
        return False
    msg = f"{path}|{exp}".encode()
    expected = hmac.new(_sign_key(), msg, hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(expected, sig)


def require_valid_signature(request: Request) -> None:
    """Dependency for report download routes — verify ?exp=&sig= against the raw path."""
    exp_s = request.query_params.get("exp")
    sig = request.query_params.get("sig")
    if not exp_s or not sig:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Missing signature")
    try:
        exp = int(exp_s)
    except ValueError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Bad signature")
    if not verify_signed_path(request.url.path, exp, sig):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired signature")
