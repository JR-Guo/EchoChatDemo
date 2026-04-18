import json
import asyncio

import httpx
import respx
from fastapi.testclient import TestClient

from tests.fixtures.make_dicom import make_still_dicom


class FakeEngine:
    def __init__(self, text="Aortic Valve: normal. Summary: ok."):
        self.text = text
    async def infer(self, system, query, images, videos):
        return self.text


def _client(monkeypatch, tmp_path, engine=None):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    monkeypatch.setenv("ECHOCHAT_SKIP_MODEL", "1")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.main import create_app
    from app.routers.tasks import get_engine
    app = create_app()
    if engine is not None:
        app.dependency_overrides[get_engine] = lambda: engine
    c = TestClient(app)
    c.post("/login", data={"password": "pw"})
    return c


@respx.mock
def test_start_report_and_stream(monkeypatch, tmp_path):
    engine = FakeEngine(
        "\n".join([
            f"{s}: ok." for s in
            ["Aortic Valve", "Atria", "Great Vessels", "Left Ventricle",
             "Mitral Valve", "Pericardium Pleural", "Pulmonic Valve",
             "Right Ventricle", "Tricuspid Valve", "Summary"]
        ])
    )
    c = _client(monkeypatch, tmp_path, engine=engine)
    sid = c.post("/api/study").json()["study_id"]

    src = make_still_dicom(tmp_path / "a.dcm")
    with src.open("rb") as f:
        c.post(f"/api/study/{sid}/upload",
               files={"file": ("a.dcm", f, "application/octet-stream")})

    respx.post("http://127.0.0.1:8995/classify").mock(
        return_value=httpx.Response(200, json={"class_name": "Apical 4C 2D", "confidence": 0.9})
    )
    with c.stream("GET", f"/api/study/{sid}/process") as r:
        _ = b"".join(r.iter_bytes())

    tid = c.post(f"/api/study/{sid}/task/report", json={}).json()["task_id"]

    import time
    # Give the background task a moment to finish
    for _ in range(50):
        resp = c.get(f"/api/task/{tid}")
        body = resp.json()
        if "status" in body and body.get("status") == "done":
            break
        if "sections" in body:
            break
        time.sleep(0.05)

    result = c.get(f"/api/task/{tid}").json()
    assert result.get("status") == "done"
    assert len(result["sections"]) == 10


import time as _time
from tests.fixtures.make_dicom import make_still_dicom as _make_still_dicom


def test_patch_report_section(monkeypatch, tmp_path):
    engine = FakeEngine(
        "\n".join([f"{s}: ok." for s in [
            "Aortic Valve","Atria","Great Vessels","Left Ventricle","Mitral Valve",
            "Pericardium Pleural","Pulmonic Valve","Right Ventricle","Tricuspid Valve","Summary",
        ]])
    )
    c = _client(monkeypatch, tmp_path, engine=engine)
    sid = c.post("/api/study").json()["study_id"]
    src = _make_still_dicom(tmp_path / "a.dcm")
    with src.open("rb") as f:
        c.post(f"/api/study/{sid}/upload",
               files={"file": ("a", f, "application/octet-stream")})

    tid = c.post(f"/api/study/{sid}/task/report", json={}).json()["task_id"]
    # wait for report.json to land on disk
    for _ in range(50):
        _time.sleep(0.05)
        result = c.get(f"/api/task/{tid}").json()
        if result.get("status") == "done":
            break

    r = c.patch(f"/api/study/{sid}/report/section",
                json={"section": "Left Ventricle", "content": "EF 55 (edited)"})
    assert r.status_code == 200
    assert r.json()["edited"] is True

    from pathlib import Path
    import json as J
    p = Path(tmp_path) / "sessions" / sid / "results" / "report.json"
    assert p.exists()
    data = J.loads(p.read_text())
    lv = next(s for s in data["sections"] if s["name"] == "Left Ventricle")
    assert lv["content"] == "EF 55 (edited)"
    assert lv["edited"] is True


def test_export_pdf(monkeypatch, tmp_path):
    engine = FakeEngine(
        "\n".join([f"{s}: ok." for s in [
            "Aortic Valve","Atria","Great Vessels","Left Ventricle","Mitral Valve",
            "Pericardium Pleural","Pulmonic Valve","Right Ventricle","Tricuspid Valve","Summary",
        ]])
    )
    c = _client(monkeypatch, tmp_path, engine=engine)
    sid = c.post("/api/study").json()["study_id"]
    src = make_still_dicom(tmp_path / "a.dcm")
    with src.open("rb") as f:
        c.post(f"/api/study/{sid}/upload", files={"file": ("a", f, "application/octet-stream")})

    tid = c.post(f"/api/study/{sid}/task/report", json={}).json()["task_id"]
    import time
    for _ in range(60):
        time.sleep(0.05)
        status = c.get(f"/api/task/{tid}").json()
        if status.get("status") == "done":
            break

    r = c.get(f"/api/study/{sid}/report/export?format=pdf")
    assert r.status_code == 200
    assert r.content[:4] == b"%PDF"
    r = c.get(f"/api/study/{sid}/report/export?format=docx")
    assert r.status_code == 200
    assert r.content[:2] == b"PK"
