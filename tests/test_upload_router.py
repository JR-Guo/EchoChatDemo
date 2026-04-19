from fastapi.testclient import TestClient

from tests.fixtures.make_dicom import make_still_dicom, make_non_dicom


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app
    c = TestClient(create_app())
    c.post("/login", data={"password": "pw"})
    return c


def test_create_study_returns_id(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/study")
    assert r.status_code == 200
    assert isinstance(r.json()["study_id"], str) and len(r.json()["study_id"]) >= 8


def test_upload_dicom_accepted(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    sid = c.post("/api/study").json()["study_id"]
    src = make_still_dicom(tmp_path / "real.dcm")
    with src.open("rb") as f:
        r = c.post(f"/api/study/{sid}/upload", files={"file": ("weird.png", f, "application/octet-stream")})
    assert r.status_code == 200
    j = r.json()
    assert j["kind"] == "dicom"
    assert j["filename"] == "weird.png"
    assert j["file_id"]


def test_upload_rejects_non_dicom(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    sid = c.post("/api/study").json()["study_id"]
    src = make_non_dicom(tmp_path / "not.dcm")
    with src.open("rb") as f:
        r = c.post(f"/api/study/{sid}/upload", files={"file": ("x.dcm", f, "application/octet-stream")})
    assert r.status_code == 400
    assert "DICOM" in r.json()["detail"]


import respx
import httpx
from tests.fixtures.make_dicom import make_cine_dicom


@respx.mock
def test_process_sse_emits_events(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    sid = c.post("/api/study").json()["study_id"]

    src = make_cine_dicom(tmp_path / "cine.dcm", frames=3, shape=(16, 16))
    with src.open("rb") as f:
        c.post(f"/api/study/{sid}/upload",
               files={"file": ("weird_cine", f, "application/octet-stream")})

    respx.post("http://127.0.0.1:8996/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"classification": {"original_view_name": "Apical 4C 2D", "confidence": 0.8}})
    )

    with c.stream("GET", f"/api/study/{sid}/process") as r:
        body = b"".join(r.iter_bytes())

    text = body.decode()
    assert "event: phase" in text
    assert "event: clip" in text
    assert "event: done" in text
