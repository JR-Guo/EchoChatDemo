from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, URLSafeSerializer
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings


_COOKIE_NAME = "echochat_session"
_PUBLIC_PATHS = {"/login", "/logout", "/healthz", "/static", "/favicon.ico"}


def _serializer() -> URLSafeSerializer:
    return URLSafeSerializer(get_settings().session_secret, salt="echochat-auth")


def is_authenticated(request: Request) -> bool:
    token = request.cookies.get(_COOKIE_NAME)
    if not token:
        return False
    try:
        data = _serializer().loads(token)
    except BadSignature:
        return False
    return data.get("ok") is True


class RequireAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if any(path == p or path.startswith(p + "/") for p in _PUBLIC_PATHS):
            return await call_next(request)
        if path == "/" or is_authenticated(request):
            return await call_next(request)
        return RedirectResponse("/login", status_code=303)


router = APIRouter()


@router.post("/login")
def login(password: str = Form(...)):
    if password != get_settings().shared_password:
        raise HTTPException(status_code=401, detail="Invalid password.")
    token = _serializer().dumps({"ok": True})
    resp = RedirectResponse("/home", status_code=303)
    resp.set_cookie(
        _COOKIE_NAME, token, httponly=True, samesite="lax", max_age=24 * 3600
    )
    return resp


@router.post("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(_COOKIE_NAME)
    return resp
