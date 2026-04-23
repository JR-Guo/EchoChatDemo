"""Tests for /api/mobile/v1/auth/* and the Bearer dependency."""
from __future__ import annotations

import time

import jwt
import pytest
from fastapi.testclient import TestClient


def _make_client(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    monkeypatch.setenv("ECHOCHAT_SKIP_MODEL", "1")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app
    return TestClient(create_app())


def test_login_requires_password(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/api/mobile/v1/auth/login", json={})
    assert r.status_code == 422


def test_login_wrong_password_rejected(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/api/mobile/v1/auth/login", json={"password": "nope"})
    assert r.status_code == 401


def test_login_correct_password_returns_token_pair(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/api/mobile/v1/auth/login", json={"password": "pw"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"access", "refresh"}
    assert body["access"] and body["refresh"]
    # Payloads contain the right typ
    access_payload = jwt.decode(body["access"], "x" * 32, algorithms=["HS256"])
    refresh_payload = jwt.decode(body["refresh"], "x" * 32, algorithms=["HS256"])
    assert access_payload["typ"] == "access"
    assert refresh_payload["typ"] == "refresh"
    # Access TTL < refresh TTL
    assert refresh_payload["exp"] - refresh_payload["iat"] > access_payload["exp"] - access_payload["iat"]


def test_refresh_returns_new_access(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/api/mobile/v1/auth/login", json={"password": "pw"})
    tokens = r.json()
    r2 = c.post("/api/mobile/v1/auth/refresh", json={"refresh": tokens["refresh"]})
    assert r2.status_code == 200
    assert "access" in r2.json()
    # New access is different from original (different jti)
    assert r2.json()["access"] != tokens["access"]


def test_refresh_rejects_access_token(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/api/mobile/v1/auth/login", json={"password": "pw"})
    tokens = r.json()
    # Presenting access as refresh must fail
    r2 = c.post("/api/mobile/v1/auth/refresh", json={"refresh": tokens["access"]})
    assert r2.status_code == 401


def test_refresh_rejects_garbage(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/api/mobile/v1/auth/refresh", json={"refresh": "not.a.token"})
    assert r.status_code == 401


def test_refresh_rejects_expired(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    # Forge an expired refresh token with the right secret
    expired = jwt.encode(
        {"typ": "refresh", "iat": int(time.time()) - 100000, "exp": int(time.time()) - 100, "jti": "x"},
        "x" * 32, algorithm="HS256",
    )
    r = c.post("/api/mobile/v1/auth/refresh", json={"refresh": expired})
    assert r.status_code == 401


def test_logout_is_always_204(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/api/mobile/v1/auth/logout", json={"refresh": "anything"})
    assert r.status_code == 204


def test_bearer_required_on_studies(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies")
    assert r.status_code == 401
    assert "Bearer" in r.headers.get("www-authenticate", "")


def test_bearer_rejects_refresh_token(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    tokens = c.post("/api/mobile/v1/auth/login", json={"password": "pw"}).json()
    r = c.get(
        "/api/mobile/v1/studies",
        headers={"Authorization": f"Bearer {tokens['refresh']}"},
    )
    # Refresh-typed token must not be accepted as access
    assert r.status_code == 401


def test_bearer_accepts_valid_access(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    tokens = c.post("/api/mobile/v1/auth/login", json={"password": "pw"}).json()
    r = c.get(
        "/api/mobile/v1/studies",
        headers={"Authorization": f"Bearer {tokens['access']}"},
    )
    assert r.status_code == 200
    assert "items" in r.json()


def test_mobile_routes_not_redirected_to_login(monkeypatch, tmp_path):
    """Mobile paths must not be caught by the web-app cookie middleware."""
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/api/mobile/v1/auth/login", json={"password": "pw"}, follow_redirects=False)
    # Not a 303 redirect to /login
    assert r.status_code == 200
