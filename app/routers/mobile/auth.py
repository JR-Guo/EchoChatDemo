from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel

from app.config import get_settings
from app.mobile_auth import (
    decode_token,
    encode_access_token,
    encode_refresh_token,
)


router = APIRouter(tags=["mobile-auth"])


class LoginBody(BaseModel):
    password: str


class LoginResponse(BaseModel):
    access: str
    refresh: str


class RefreshBody(BaseModel):
    refresh: str


class RefreshResponse(BaseModel):
    access: str


@router.post("/auth/login", response_model=LoginResponse)
def login(body: LoginBody) -> LoginResponse:
    if body.password != get_settings().shared_password:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid password")
    return LoginResponse(
        access=encode_access_token(),
        refresh=encode_refresh_token(),
    )


@router.post("/auth/refresh", response_model=RefreshResponse)
def refresh(body: RefreshBody) -> RefreshResponse:
    decode_token(body.refresh, expected_typ="refresh")
    return RefreshResponse(access=encode_access_token())


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(body: RefreshBody) -> Response:
    # M1: no revocation list — token expiry is the only invalidation path.
    # Client clears SecureStore regardless of this response.
    return Response(status_code=status.HTTP_204_NO_CONTENT)
