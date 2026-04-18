from fastapi.testclient import TestClient


def _make_client(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app
    return TestClient(create_app())


def test_requires_login_to_access_home(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/home", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_login_wrong_password_rejected(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/login", data={"password": "wrong"}, follow_redirects=False)
    assert r.status_code == 401


def test_login_correct_password_sets_cookie_and_redirects(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/login", data={"password": "pw"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/home"
    assert "set-cookie" in r.headers


def test_authed_request_reaches_home(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    c.post("/login", data={"password": "pw"})
    r = c.get("/home")
    assert r.status_code == 200


def test_logout_clears_cookie(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    c.post("/login", data={"password": "pw"})
    c.post("/logout")
    r = c.get("/home", follow_redirects=False)
    assert r.status_code == 303


def test_healthz_is_public(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/healthz")
    assert r.status_code == 200


def test_login_page_renders_english_form(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/login")
    assert r.status_code == 200
    assert "Sign in" in r.text
    assert "password" in r.text.lower()
    assert "<html lang=\"en\"" in r.text


def test_home_page_has_task_cards(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    c.post("/login", data={"password": "pw"})
    r = c.get("/home")
    assert r.status_code == 200
    assert "Report Generation" in r.text
    assert "Measurement" in r.text
    assert "Disease Diagnosis" in r.text
    assert "Visual Question Answering" in r.text
    assert "Start new study" in r.text
