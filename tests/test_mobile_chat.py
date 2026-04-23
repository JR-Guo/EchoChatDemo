"""Tests for /api/mobile/v1/studies/{id}/messages and /chat (SSE)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi.testclient import TestClient

from app.models.study import Study, TasksAvailability


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


def _make_study(tmp_path, sid: str) -> None:
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


class _FakeEngine:
    """Stand-in for EchoChatEngine that yields a fixed list of deltas."""

    def __init__(self, deltas: list[str] | None = None):
        self.deltas = deltas if deltas is not None else ["Hello", " world", "!"]
        self.calls: list[dict] = []
        self.raise_exc: BaseException | None = None

    async def infer_stream(self, *, system: str, query: str, images, videos) -> AsyncIterator[str]:
        self.calls.append({"system": system, "query": query, "images": list(images), "videos": list(videos)})
        if self.raise_exc is not None:
            raise self.raise_exc
        for d in self.deltas:
            yield d


def test_messages_empty_when_no_chat(monkeypatch, tmp_path):
    _make_study(tmp_path, "sa")
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies/sa/messages", headers=_auth(c))
    assert r.status_code == 200
    assert r.json() == []


def test_messages_requires_auth(monkeypatch, tmp_path):
    _make_study(tmp_path, "sa")
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies/sa/messages")
    assert r.status_code == 401


def test_messages_404_for_missing_study(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/api/mobile/v1/studies/nope/messages", headers=_auth(c))
    assert r.status_code == 404


def test_chat_streams_tokens_and_persists(monkeypatch, tmp_path):
    _make_study(tmp_path, "sb")
    engine = _FakeEngine(deltas=["Hello", " clinician", "."])
    c = _make_client(monkeypatch, tmp_path, engine=engine)

    with c.stream(
        "POST", "/api/mobile/v1/studies/sb/chat",
        headers=_auth(c),
        json={"content": "What view is this?"},
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        frames = [line for line in r.iter_lines() if line.startswith("data:")]

    events = [json.loads(f.split("data:", 1)[1].strip()) for f in frames]
    token_deltas = [e["delta"] for e in events if e["type"] == "token"]
    done_events = [e for e in events if e["type"] == "done"]

    assert token_deltas == ["Hello", " clinician", "."]
    assert len(done_events) == 1
    assert done_events[0]["messageId"]

    # History now has exactly [user, assistant]
    history = c.get("/api/mobile/v1/studies/sb/messages", headers=_auth(c)).json()
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "What view is this?"
    assert history[1]["content"] == "Hello clinician."
    assert history[1]["id"] == done_events[0]["messageId"]


def test_chat_multi_turn_feeds_history_to_engine(monkeypatch, tmp_path):
    _make_study(tmp_path, "sc")
    engine = _FakeEngine(deltas=["First answer."])
    c = _make_client(monkeypatch, tmp_path, engine=engine)

    with c.stream("POST", "/api/mobile/v1/studies/sc/chat",
                  headers=_auth(c), json={"content": "Q1"}) as r:
        list(r.iter_lines())

    engine.deltas = ["Second answer."]
    with c.stream("POST", "/api/mobile/v1/studies/sc/chat",
                  headers=_auth(c), json={"content": "Q2"}) as r:
        list(r.iter_lines())

    # First call: query is plain (no prior history)
    assert engine.calls[0]["query"] == "Q1"
    # Second call: query includes the first turn as prefix
    assert "Previous conversation" in engine.calls[1]["query"]
    assert "Doctor: Q1" in engine.calls[1]["query"]
    assert "Assistant: First answer." in engine.calls[1]["query"]
    assert engine.calls[1]["query"].rstrip().endswith("Doctor: Q2\nAssistant:")


def test_chat_engine_error_reported_as_error_frame(monkeypatch, tmp_path):
    _make_study(tmp_path, "sd")
    engine = _FakeEngine()
    engine.raise_exc = RuntimeError("CUDA OOM")
    c = _make_client(monkeypatch, tmp_path, engine=engine)

    with c.stream("POST", "/api/mobile/v1/studies/sd/chat",
                  headers=_auth(c), json={"content": "hi"}) as r:
        frames = [json.loads(l.split("data:", 1)[1].strip()) for l in r.iter_lines() if l.startswith("data:")]

    types = [e["type"] for e in frames]
    assert "error" in types
    err = [e for e in frames if e["type"] == "error"][0]
    assert "CUDA OOM" in err["message"]
    # User message was still persisted (placeholder assistant may be empty)
    history = c.get("/api/mobile/v1/studies/sd/messages", headers=_auth(c)).json()
    roles = [m["role"] for m in history]
    assert roles[0] == "user"


def test_chat_validates_body(monkeypatch, tmp_path):
    _make_study(tmp_path, "se")
    c = _make_client(monkeypatch, tmp_path, engine=_FakeEngine())
    r = c.post("/api/mobile/v1/studies/se/chat", headers=_auth(c), json={"content": ""})
    assert r.status_code == 422


def test_chat_404_for_missing_study(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path, engine=_FakeEngine())
    with c.stream("POST", "/api/mobile/v1/studies/nope/chat",
                  headers=_auth(c), json={"content": "hi"}) as r:
        assert r.status_code == 404


def test_chat_requires_auth(monkeypatch, tmp_path):
    _make_study(tmp_path, "sf")
    c = _make_client(monkeypatch, tmp_path, engine=_FakeEngine())
    r = c.post("/api/mobile/v1/studies/sf/chat", json={"content": "hi"})
    assert r.status_code == 401
