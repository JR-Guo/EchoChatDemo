"""Tests for /api/mobile/v1/studies/{id}/report and signed downloads."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.models.study import Study, TasksAvailability


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


def _make_study(tmp_path, sid: str) -> "tuple[object, object]":
    root = tmp_path / "sessions" / sid
    for sub in ("raw", "converted", "results", "exports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    study = Study(
        study_id=sid,
        created_at=datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc),
        clips=[],
        tasks=TasksAvailability(report=False, measurement=False, disease=False, vqa=False, missing_groups=[]),
    )
    (root / "meta.json").write_text(study.model_dump_json(indent=2))
    return root, study


def test_get_report_404_when_none_exists(monkeypatch, tmp_path):
    _make_study(tmp_path, "sa")
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies/sa/report", headers=_auth(c))
    assert r.status_code == 404


def test_get_report_returns_signed_urls(monkeypatch, tmp_path):
    root, _ = _make_study(tmp_path, "sb")
    (root / "exports" / "report.pdf").write_bytes(b"%PDF-stub")
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies/sb/report", headers=_auth(c))
    assert r.status_code == 200
    body = r.json()
    assert body["studyId"] == "sb"
    assert body["status"] == "ready"
    assert body["pdfUrl"].startswith("/api/mobile/v1/studies/sb/report/pdf?")
    assert "exp=" in body["pdfUrl"] and "sig=" in body["pdfUrl"]
    assert body["docxUrl"] is None


def test_post_report_pending_when_not_yet_generated(monkeypatch, tmp_path):
    _make_study(tmp_path, "sc")
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/api/mobile/v1/studies/sc/report", headers=_auth(c))
    assert r.status_code == 202
    assert r.json()["status"] == "pending"


def test_signed_download_rejects_tampered_sig(monkeypatch, tmp_path):
    root, _ = _make_study(tmp_path, "sd")
    (root / "exports" / "report.pdf").write_bytes(b"%PDF-stub")
    c = _make_client(monkeypatch, tmp_path)
    signed = c.get("/api/mobile/v1/studies/sd/report", headers=_auth(c)).json()["pdfUrl"]
    tampered = signed.replace("sig=", "sig=0")
    # Without the Bearer header, signature is the only gate
    r = c.get(tampered)
    assert r.status_code == 401


def test_signed_download_works(monkeypatch, tmp_path):
    root, _ = _make_study(tmp_path, "se")
    (root / "exports" / "report.pdf").write_bytes(b"%PDF-stub")
    c = _make_client(monkeypatch, tmp_path)
    signed = c.get("/api/mobile/v1/studies/se/report", headers=_auth(c)).json()["pdfUrl"]
    r = c.get(signed)  # no bearer — signature alone authorizes
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content == b"%PDF-stub"


def test_signed_download_rejects_missing_query(monkeypatch, tmp_path):
    root, _ = _make_study(tmp_path, "sf")
    (root / "exports" / "report.pdf").write_bytes(b"%PDF-stub")
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies/sf/report/pdf")
    assert r.status_code == 401
