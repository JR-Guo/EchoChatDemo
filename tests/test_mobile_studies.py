"""Tests for /api/mobile/v1/studies."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.models.study import Clip, Study, TasksAvailability


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


def _make_study(tmp_path, sid: str, created_at: datetime, clips: list[Clip] | None = None):
    root = tmp_path / "sessions" / sid
    for sub in ("raw", "converted", "results", "exports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    study = Study(
        study_id=sid,
        created_at=created_at,
        clips=clips or [],
        tasks=TasksAvailability(report=False, measurement=False, disease=False, vqa=False, missing_groups=[]),
    )
    (root / "meta.json").write_text(study.model_dump_json(indent=2))
    return root


def test_list_empty_when_no_studies(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies", headers=_auth(c))
    assert r.status_code == 200
    assert r.json() == {"items": [], "nextCursor": None}


def test_list_returns_studies_newest_first(monkeypatch, tmp_path):
    _make_study(tmp_path, "aaaa", datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc))
    _make_study(tmp_path, "bbbb", datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc))
    _make_study(tmp_path, "cccc", datetime(2026, 4, 22, 10, 0, tzinfo=timezone.utc))

    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies", headers=_auth(c))
    assert r.status_code == 200
    ids = [it["id"] for it in r.json()["items"]]
    assert ids == ["cccc", "bbbb", "aaaa"]


def test_list_cursor_pagination(monkeypatch, tmp_path):
    for i in range(5):
        _make_study(tmp_path, f"s{i}", datetime(2026, 4, 20 + i, 10, 0, tzinfo=timezone.utc))
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies?limit=2", headers=_auth(c))
    body = r.json()
    assert len(body["items"]) == 2
    assert body["nextCursor"]

    r2 = c.get(f"/api/mobile/v1/studies?limit=2&cursor={body['nextCursor']}", headers=_auth(c))
    body2 = r2.json()
    assert len(body2["items"]) == 2
    # distinct from first page
    assert {it["id"] for it in body2["items"]} & {it["id"] for it in body["items"]} == set()


def test_list_bad_cursor_is_400(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies?cursor=not-base64", headers=_auth(c))
    assert r.status_code == 400


def test_list_skips_corrupt_meta(monkeypatch, tmp_path):
    _make_study(tmp_path, "good", datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc))
    bad_root = tmp_path / "sessions" / "bad"
    bad_root.mkdir(parents=True)
    (bad_root / "meta.json").write_text("not valid json")
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies", headers=_auth(c))
    ids = [it["id"] for it in r.json()["items"]]
    assert ids == ["good"]


def test_list_skips_soft_deleted(monkeypatch, tmp_path):
    root = _make_study(tmp_path, "keep", datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc))
    gone = _make_study(tmp_path, "gone", datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc))
    (gone / ".deleted").write_text("2026-04-22")
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies", headers=_auth(c))
    ids = [it["id"] for it in r.json()["items"]]
    assert ids == ["keep"]


def test_get_study_returns_detail(monkeypatch, tmp_path):
    clips = [
        Clip(
            file_id="f1", original_filename="a.dcm", kind="dicom",
            raw_path="/raw/a.dcm", view="A4C", confidence=0.91, is_video=True,
            converted_path="/nonexistent.mp4",
        ),
    ]
    _make_study(tmp_path, "sx", datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc), clips=clips)
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies/sx", headers=_auth(c))
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "sx"
    assert body["status"] == "ready"
    assert body["viewCount"] == 1
    assert len(body["views"]) == 1
    v = body["views"][0]
    assert v["name"] == "A4C"
    assert 0.90 < v["confidence"] < 0.92
    assert v["thumbnailUrl"].startswith("/api/mobile/v1/studies/sx/clips/f1/thumbnail?")
    assert v["mp4Url"] and v["mp4Url"].startswith("/api/mobile/v1/studies/sx/clips/f1/video?")


def test_get_study_404(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies/missing", headers=_auth(c))
    assert r.status_code == 404


def test_delete_soft_deletes(monkeypatch, tmp_path):
    root = _make_study(tmp_path, "sz", datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc))
    c = _make_client(monkeypatch, tmp_path)
    r = c.delete("/api/mobile/v1/studies/sz", headers=_auth(c))
    assert r.status_code == 204
    assert (root / ".deleted").exists()
    # No longer in the list
    r2 = c.get("/api/mobile/v1/studies", headers=_auth(c))
    assert r2.json()["items"] == []
    # But directory still on disk for audit
    assert (root / "meta.json").exists()


def test_delete_404(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.delete("/api/mobile/v1/studies/missing", headers=_auth(c))
    assert r.status_code == 404
