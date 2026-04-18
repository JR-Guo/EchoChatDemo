import httpx
import respx
from fastapi.testclient import TestClient

from tests.fixtures.make_dicom import make_still_dicom


def _authed_client(monkeypatch, tmp_path):
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


def _new_study_with_clip(c, tmp_path):
    sid = c.post("/api/study").json()["study_id"]
    src = make_still_dicom(tmp_path / "s.dcm")
    with src.open("rb") as f:
        c.post(f"/api/study/{sid}/upload",
               files={"file": ("s.dcm", f, "application/octet-stream")})
    return sid


@respx.mock
def test_get_study_returns_tasks_availability(monkeypatch, tmp_path):
    c = _authed_client(monkeypatch, tmp_path)
    sid = _new_study_with_clip(c, tmp_path)
    respx.post("http://127.0.0.1:8995/classify").mock(
        return_value=httpx.Response(200, json={"class_name": "Apical 4C 2D", "confidence": 0.7})
    )
    with c.stream("GET", f"/api/study/{sid}/process") as r:
        _ = b"".join(r.iter_bytes())
    body = c.get(f"/api/study/{sid}").json()
    assert body["tasks"]["report"] is True
    assert isinstance(body["tasks"]["missing_groups"], list)


def test_patch_clip_user_view(monkeypatch, tmp_path):
    c = _authed_client(monkeypatch, tmp_path)
    sid = _new_study_with_clip(c, tmp_path)
    fid = c.get(f"/api/study/{sid}").json()["clips"][0]["file_id"]

    r = c.patch(f"/api/study/{sid}/clip/{fid}", json={"user_view": "Apical 2C 2D"})
    assert r.status_code == 200
    assert r.json()["user_view"] == "Apical 2C 2D"


def test_patch_clip_rejects_unknown_view(monkeypatch, tmp_path):
    c = _authed_client(monkeypatch, tmp_path)
    sid = _new_study_with_clip(c, tmp_path)
    fid = c.get(f"/api/study/{sid}").json()["clips"][0]["file_id"]

    r = c.patch(f"/api/study/{sid}/clip/{fid}", json={"user_view": "Not A View"})
    assert r.status_code == 400


def test_delete_clip(monkeypatch, tmp_path):
    c = _authed_client(monkeypatch, tmp_path)
    sid = _new_study_with_clip(c, tmp_path)
    fid = c.get(f"/api/study/{sid}").json()["clips"][0]["file_id"]
    assert c.delete(f"/api/study/{sid}/clip/{fid}").status_code == 204
    assert c.get(f"/api/study/{sid}").json()["clips"] == []
