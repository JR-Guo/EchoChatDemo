"""Tests for /api/mobile/v1/uploads (tus) and POST /studies (create from upload)."""
from __future__ import annotations

import base64
from pathlib import Path

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


def _auth(client: TestClient) -> dict:
    tokens = client.post("/api/mobile/v1/auth/login", json={"password": "pw"}).json()
    return {"Authorization": f"Bearer {tokens['access']}"}


def _tus(extra: dict | None = None) -> dict:
    h = {"Tus-Resumable": "1.0.0"}
    if extra:
        h.update(extra)
    return h


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


# ---- tus protocol ---------------------------------------------------------

def test_options_advertises_extensions(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.options("/api/mobile/v1/uploads")
    assert r.status_code == 204
    assert r.headers["tus-resumable"] == "1.0.0"
    assert "creation" in r.headers["tus-extension"]
    assert "termination" in r.headers["tus-extension"]
    assert int(r.headers["tus-max-size"]) > 0


def test_create_requires_auth(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/api/mobile/v1/uploads", headers=_tus({"Upload-Length": "10"}))
    assert r.status_code == 401


def test_create_requires_tus_resumable(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    h.update({"Upload-Length": "10"})
    r = c.post("/api/mobile/v1/uploads", headers=h)
    assert r.status_code == 412


def test_create_upload_returns_location(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    h.update(_tus({
        "Upload-Length": "11",
        "Upload-Metadata": f"filename {_b64('hello.txt')},filetype {_b64('text/plain')}",
    }))
    r = c.post("/api/mobile/v1/uploads", headers=h)
    assert r.status_code == 201
    loc = r.headers["location"]
    assert loc.startswith("/api/mobile/v1/uploads/")
    assert r.headers["upload-offset"] == "0"


def test_head_before_upload_offset_zero(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    create = c.post("/api/mobile/v1/uploads", headers={**h, **_tus({"Upload-Length": "5"})})
    uid = create.headers["location"].rsplit("/", 1)[-1]
    r = c.head(f"/api/mobile/v1/uploads/{uid}", headers={**h, **_tus()})
    assert r.status_code == 200
    assert r.headers["upload-offset"] == "0"
    assert r.headers["upload-length"] == "5"


def test_patch_chunks_advance_offset(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    create = c.post("/api/mobile/v1/uploads", headers={**h, **_tus({"Upload-Length": "11"})})
    uid = create.headers["location"].rsplit("/", 1)[-1]

    patch_h = {**h, **_tus({"Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"})}
    r1 = c.patch(f"/api/mobile/v1/uploads/{uid}", headers=patch_h, content=b"hello ")
    assert r1.status_code == 204
    assert r1.headers["upload-offset"] == "6"

    patch_h2 = {**h, **_tus({"Upload-Offset": "6", "Content-Type": "application/offset+octet-stream"})}
    r2 = c.patch(f"/api/mobile/v1/uploads/{uid}", headers=patch_h2, content=b"world")
    assert r2.status_code == 204
    assert r2.headers["upload-offset"] == "11"


def test_patch_rejects_offset_mismatch(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    create = c.post("/api/mobile/v1/uploads", headers={**h, **_tus({"Upload-Length": "11"})})
    uid = create.headers["location"].rsplit("/", 1)[-1]

    patch_h = {**h, **_tus({"Upload-Offset": "3", "Content-Type": "application/offset+octet-stream"})}
    r = c.patch(f"/api/mobile/v1/uploads/{uid}", headers=patch_h, content=b"nope")
    assert r.status_code == 409
    assert r.headers["upload-offset"] == "0"


def test_patch_wrong_content_type(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    create = c.post("/api/mobile/v1/uploads", headers={**h, **_tus({"Upload-Length": "5"})})
    uid = create.headers["location"].rsplit("/", 1)[-1]
    r = c.patch(
        f"/api/mobile/v1/uploads/{uid}",
        headers={**h, **_tus({"Upload-Offset": "0", "Content-Type": "application/json"})},
        content=b"hello",
    )
    assert r.status_code == 415


def test_delete_removes_upload(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    create = c.post("/api/mobile/v1/uploads", headers={**h, **_tus({"Upload-Length": "5"})})
    uid = create.headers["location"].rsplit("/", 1)[-1]
    r = c.delete(f"/api/mobile/v1/uploads/{uid}", headers={**h, **_tus()})
    assert r.status_code == 204
    # subsequent HEAD → 404
    r2 = c.head(f"/api/mobile/v1/uploads/{uid}", headers={**h, **_tus()})
    assert r2.status_code == 404


# ---- POST /studies (create from upload) -----------------------------------

# PNG: 1x1 red pixel (http://www.schaik.com/pngsuite/)
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c62fccfc0c00000000500013073c04d0000000049454e44ae426082"
)


def _upload_bytes(c: TestClient, auth: dict, payload: bytes, filename: str, filetype: str) -> str:
    hdrs = {**auth, **_tus({
        "Upload-Length": str(len(payload)),
        "Upload-Metadata": f"filename {_b64(filename)},filetype {_b64(filetype)}",
    })}
    create = c.post("/api/mobile/v1/uploads", headers=hdrs)
    uid = create.headers["location"].rsplit("/", 1)[-1]
    patch_h = {**auth, **_tus({"Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"})}
    r = c.patch(f"/api/mobile/v1/uploads/{uid}", headers=patch_h, content=payload)
    assert r.status_code == 204
    return uid


def test_create_study_from_image(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    uid = _upload_bytes(c, h, _PNG_1x1, "test.png", "image/png")

    r = c.post("/api/mobile/v1/studies", headers=h, json={"uploadId": uid})
    assert r.status_code == 201, r.text
    detail = r.json()
    assert detail["viewCount"] == 1
    assert detail["status"] in ("queued", "processing", "ready")
    # Files landed on disk
    sid = detail["id"]
    study_dir = tmp_path / "sessions" / sid
    assert (study_dir / "meta.json").exists()
    converted = list((study_dir / "converted").glob("*.png"))
    assert converted, "image should have been installed into converted/"


def test_create_study_from_mp4_stub(monkeypatch, tmp_path):
    # Minimal MP4 ftyp box that our detect_kind accepts (content-only sniff,
    # no decode). _extract_video_middle_frame may fail on this stub; that's
    # caught and thumbnail is skipped — clip still persists.
    mp4_stub = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 100
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    uid = _upload_bytes(c, h, mp4_stub, "clip.mp4", "video/mp4")

    r = c.post("/api/mobile/v1/studies", headers=h, json={"uploadId": uid})
    assert r.status_code == 201, r.text
    detail = r.json()
    assert detail["viewCount"] == 1


def test_create_study_unsupported_type(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    uid = _upload_bytes(c, h, b"just some random text", "data.txt", "text/plain")
    r = c.post("/api/mobile/v1/studies", headers=h, json={"uploadId": uid})
    assert r.status_code == 422


def test_create_study_unknown_upload_id(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    r = c.post("/api/mobile/v1/studies", headers=h, json={"uploadId": "deadbeef"})
    assert r.status_code == 404


def test_create_study_upload_not_complete(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    # Create without patching any bytes
    create = c.post("/api/mobile/v1/uploads", headers={**h, **_tus({"Upload-Length": "5"})})
    uid = create.headers["location"].rsplit("/", 1)[-1]
    r = c.post("/api/mobile/v1/studies", headers=h, json={"uploadId": uid})
    assert r.status_code == 409


def test_create_study_requires_auth(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/api/mobile/v1/studies", json={"uploadId": "x"})
    assert r.status_code == 401


def test_upload_consumed_after_study_creation(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    uid = _upload_bytes(c, h, _PNG_1x1, "test.png", "image/png")
    assert c.post("/api/mobile/v1/studies", headers=h, json={"uploadId": uid}).status_code == 201
    # Upload directory is cleaned
    r = c.head(f"/api/mobile/v1/uploads/{uid}", headers={**h, **_tus()})
    assert r.status_code == 404
