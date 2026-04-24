"""Tests for /api/mobile/v1/studies/{id}/report and signed downloads."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi.testclient import TestClient

from app.models.study import Clip, Study, TasksAvailability


def _make_client(monkeypatch, tmp_path, engine=None):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    monkeypatch.setenv("ECHOCHAT_SKIP_MODEL", "1")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app
    app = create_app()
    if engine is not None:
        from app.routers.tasks import set_engine
        set_engine(engine)
    return TestClient(app)


def _auth(client: TestClient) -> dict:
    tokens = client.post("/api/mobile/v1/auth/login", json={"password": "pw"}).json()
    return {"Authorization": f"Bearer {tokens['access']}"}


def _make_study(tmp_path, sid: str, clips: list[Clip] | None = None):
    root = tmp_path / "sessions" / sid
    for sub in ("raw", "converted", "results", "exports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    study = Study(
        study_id=sid,
        created_at=datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc),
        clips=clips or [],
        tasks=TasksAvailability(report=False, measurement=False, disease=False, vqa=False, missing_groups=[]),
    )
    (root / "meta.json").write_text(study.model_dump_json(indent=2))
    return root, study


def _write_report_json(study_root, sections=None):
    sections = sections or [
        {"name": "Summary", "content": "Left ventricular systolic function appears normal."},
        {"name": "Aortic Valve", "content": "Trileaflet, no stenosis."},
    ]
    data = {"status": "done", "sections": sections}
    p = study_root / "results" / "report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    return data


class _FakeEngine:
    def __init__(self, raw: str = "Summary All normal."):
        self.raw = raw

    async def infer(self, *, system: str, query: str, images, videos) -> str:
        return self.raw

    async def infer_stream(self, *, system: str, query: str, images, videos) -> AsyncIterator[str]:
        yield self.raw


def _write_real_png(path) -> None:
    """PIL-generated 8x8 red PNG — passes Image.verify() CRC checks."""
    from PIL import Image
    Image.new("RGB", (8, 8), (200, 0, 0)).save(path, "PNG")


def _attach_image_clip(tmp_path, sid: str, file_id: str = "f1") -> None:
    """Write a real PNG into the study's converted/ dir + re-save meta with a Clip pointing at it."""
    from app.models.study import Study, TasksAvailability
    root = tmp_path / "sessions" / sid
    img = root / "converted" / f"{file_id}.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    _write_real_png(img)
    clip = Clip(
        file_id=file_id, original_filename=f"{file_id}.png", kind="image",
        raw_path=str(img), converted_path=str(img), is_video=False,
    )
    study = Study(
        study_id=sid,
        created_at=datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc),
        clips=[clip],
        tasks=TasksAvailability(report=False, measurement=False, disease=False, vqa=False, missing_groups=[]),
    )
    (root / "meta.json").write_text(study.model_dump_json(indent=2))


# ---- GET /report ----------------------------------------------------------

def test_get_report_404_when_none_exists(monkeypatch, tmp_path):
    _make_study(tmp_path, "sa")
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies/sa/report", headers=_auth(c))
    assert r.status_code == 404


def test_get_report_lazy_renders_from_json(monkeypatch, tmp_path):
    root, _ = _make_study(tmp_path, "sb")
    _write_report_json(root)
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies/sb/report", headers=_auth(c))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ready"
    assert body["studyId"] == "sb"
    assert len(body["sections"]) == 2
    assert body["sections"][0]["name"] == "Summary"
    assert body["pdfUrl"] and "sig=" in body["pdfUrl"]
    assert body["docxUrl"] and "sig=" in body["docxUrl"]
    assert (root / "exports" / "report.pdf").exists()
    assert (root / "exports" / "report.docx").exists()


# ---- POST /report ---------------------------------------------------------

def test_post_report_422_when_no_clips(monkeypatch, tmp_path):
    _make_study(tmp_path, "sc")
    c = _make_client(monkeypatch, tmp_path, engine=_FakeEngine())
    r = c.post("/api/mobile/v1/studies/sc/report", headers=_auth(c))
    assert r.status_code == 422


def test_post_report_runs_engine_and_persists(monkeypatch, tmp_path):
    _make_study(tmp_path, "sd")
    _attach_image_clip(tmp_path, "sd")
    engine = _FakeEngine(raw="Summary The left ventricle is normal. Aortic Valve Trileaflet.")
    c = _make_client(monkeypatch, tmp_path, engine=engine)
    r = c.post("/api/mobile/v1/studies/sd/report", headers=_auth(c))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ready"
    assert any(s["name"] == "Summary" for s in body["sections"])
    root = tmp_path / "sessions" / "sd"
    assert (root / "results" / "report.json").exists()
    assert (root / "exports" / "report.pdf").exists()
    assert (root / "exports" / "report.docx").exists()


def test_post_report_404_unknown_study(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path, engine=_FakeEngine())
    r = c.post("/api/mobile/v1/studies/nope/report", headers=_auth(c))
    assert r.status_code == 404


def test_post_report_requires_auth(monkeypatch, tmp_path):
    _make_study(tmp_path, "se")
    c = _make_client(monkeypatch, tmp_path, engine=_FakeEngine())
    r = c.post("/api/mobile/v1/studies/se/report")
    assert r.status_code == 401


# ---- Signed PDF/DOCX downloads -------------------------------------------

def test_signed_download_works(monkeypatch, tmp_path):
    root, _ = _make_study(tmp_path, "sf")
    _write_report_json(root)
    c = _make_client(monkeypatch, tmp_path)
    pdf_url = c.get("/api/mobile/v1/studies/sf/report", headers=_auth(c)).json()["pdfUrl"]
    r = c.get(pdf_url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")


def test_signed_download_rejects_tampered_sig(monkeypatch, tmp_path):
    root, _ = _make_study(tmp_path, "sg")
    _write_report_json(root)
    c = _make_client(monkeypatch, tmp_path)
    pdf_url = c.get("/api/mobile/v1/studies/sg/report", headers=_auth(c)).json()["pdfUrl"]
    tampered = pdf_url.replace("sig=", "sig=0")
    r = c.get(tampered)
    assert r.status_code == 401


def test_signed_download_rejects_missing_query(monkeypatch, tmp_path):
    root, _ = _make_study(tmp_path, "sh")
    _write_report_json(root)
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies/sh/report/pdf")
    assert r.status_code == 401
