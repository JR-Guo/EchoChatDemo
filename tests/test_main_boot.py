from fastapi.testclient import TestClient


def test_healthz_returns_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "s" * 32)

    from app.config import get_settings
    get_settings.cache_clear()

    from app.main import create_app
    client = TestClient(create_app())

    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert "uptime_s" in body
    assert body["model_loaded"] is False
