"""Mobile chat history persistence.

One JSON file per study at `data/sessions/<sid>/mobile_messages.json`. Schema:

    [
      {"id": "<uuid-hex>",
       "role": "user" | "assistant",
       "content": "...",
       "createdAt": "2026-04-23T12:34:56Z"},
      ...
    ]

Append-only from the caller's perspective; on-disk writes are atomic
(write temp + rename) so a torn file is never visible to readers.
The same NFS-safe `mkdir` lock pattern used by `Storage` for meta.json
serializes concurrent writers.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


FILENAME = "mobile_messages.json"
LOCK_DIRNAME = "mobile_messages.lockdir"
CONTEXT_TURNS = 6  # last N messages from history fed back to the model


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _messages_path(study_root: Path) -> Path:
    return study_root / FILENAME


def _lock_path(study_root: Path) -> Path:
    return study_root / LOCK_DIRNAME


@contextmanager
def _study_chat_lock(study_root: Path) -> Iterator[None]:
    lock = _lock_path(study_root)
    deadline = time.monotonic() + 30.0
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out acquiring chat lock for {study_root}")
            time.sleep(0.02)
    try:
        yield
    finally:
        try:
            os.rmdir(lock)
        except FileNotFoundError:
            pass


def load_messages(study_root: Path) -> list[dict]:
    path = _messages_path(study_root)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text() or "[]")
    except json.JSONDecodeError:
        # Corrupt file: treat as empty rather than 500 the request.
        return []


def _write_atomic(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def append_message(
    study_root: Path,
    *,
    role: str,
    content: str,
    message_id: str | None = None,
) -> dict:
    """Append a single message; returns the persisted dict (with id/createdAt)."""
    msg = {
        "id": message_id or uuid.uuid4().hex,
        "role": role,
        "content": content,
        "createdAt": _now_iso(),
    }
    path = _messages_path(study_root)
    with _study_chat_lock(study_root):
        history = load_messages(study_root)
        history.append(msg)
        _write_atomic(path, history)
    return msg


def replace_last_assistant(study_root: Path, content: str, message_id: str) -> dict:
    """Replace the last assistant message's content + id (used after a partial stream
    was rolled into a real persistence at `done` time). Falls back to append if the
    last message is not an assistant.
    """
    path = _messages_path(study_root)
    with _study_chat_lock(study_root):
        history = load_messages(study_root)
        if history and history[-1].get("role") == "assistant":
            history[-1]["content"] = content
            history[-1]["id"] = message_id
            history[-1]["createdAt"] = _now_iso()
            msg = history[-1]
        else:
            msg = {
                "id": message_id,
                "role": "assistant",
                "content": content,
                "createdAt": _now_iso(),
            }
            history.append(msg)
        _write_atomic(path, history)
    return msg


def context_window(messages: list[dict], limit: int = CONTEXT_TURNS) -> list[dict]:
    """Return the last `limit` messages to feed back to the model as history."""
    return messages[-limit:] if limit > 0 else []
