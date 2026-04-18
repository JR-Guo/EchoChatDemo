from fastapi.testclient import TestClient


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


def test_constants_has_all_lists(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.get("/api/constants")
    assert r.status_code == 200
    body = r.json()
    assert len(body["diseases"]) == 28
    assert len(body["measurements"]) == 22
    assert len(body["report_sections"]) == 10
    assert len(body["views"]) == 38
    assert "measurements" in body["presets"]
    assert "diseases" in body["presets"]
    assert isinstance(body["presets"]["vqa_examples"], list)


def test_constants_requires_auth(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app
    c = TestClient(create_app())
    r = c.get("/api/constants", follow_redirects=False)
    assert r.status_code == 303
