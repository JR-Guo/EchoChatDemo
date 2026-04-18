import os
from pathlib import Path

import pytest


def test_settings_load_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", "/tmp/model")
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VIEW_CLASSIFIER_URL", "http://x:1")
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "s")

    from app.config import get_settings

    get_settings.cache_clear()
    s = get_settings()

    assert s.model_path == Path("/tmp/model")
    assert s.data_dir == tmp_path
    assert s.view_classifier_url == "http://x:1"
    assert s.shared_password == "pw"
    assert s.session_secret == "s"
    assert s.host == "0.0.0.0"
    assert s.port == 12345


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "s")
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))

    from app.config import get_settings

    get_settings.cache_clear()
    s = get_settings()

    assert s.view_classifier_url == "http://127.0.0.1:8995"
