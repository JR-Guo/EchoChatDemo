"""Mobile multi-turn chat. Independent from the web VQA flow:

- Persistence: `data/sessions/<sid>/mobile_messages.json` (append-only)
- Context: last `CONTEXT_TURNS` messages fed to the model as history
- Streaming: SSE, token deltas as `{"type":"token","delta":"..."}`,
  terminated by `{"type":"done","messageId":"..."}` or `{"type":"error","message":"..."}`
- Cancellation: when the client disconnects, the generator cleans up by
  persisting whatever partial text was produced under the placeholder id.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.mobile_auth import require_mobile_user
from app.routers.tasks import get_engine
from app.services import mobile_chat
from app.services.tasks.base import collect_media
from app.storage import Storage


router = APIRouter(tags=["mobile-chat"])


MOBILE_CHAT_SYSTEM = (
    "You are an expert echocardiologist assisting a physician who is reviewing "
    "an echocardiography study on a mobile device. Answer concisely, be clinical, "
    "and cite what is visible in the provided echo views when relevant. If a "
    "question cannot be answered from the given study, say so explicitly."
)


ChatRole = Literal["user", "assistant", "system"]


class ChatMessage(BaseModel):
    id: str
    role: ChatRole
    content: str
    createdAt: str


class ChatBody(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


def _require_study_root(sid: str) -> Path:
    store = _store()
    root = store.study_root(sid)
    if not root.exists() or (root / ".deleted").exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Study not found")
    return root


def _build_history_query(history: list[dict], new_question: str) -> str:
    """Flatten prior turns into a plain transcript prefix.

    The echocardiography model is VQA-style and has no native multi-turn
    template, so we feed history as text and let the system prompt frame it
    as a continuing conversation.
    """
    if not history:
        return new_question
    lines: list[str] = ["Previous conversation:"]
    for msg in history:
        tag = "Doctor" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{tag}: {msg.get('content', '').strip()}")
    lines.append("")
    lines.append(f"Doctor: {new_question}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


@router.get(
    "/studies/{sid}/messages",
    response_model=list[ChatMessage],
    dependencies=[Depends(require_mobile_user)],
)
def get_messages(sid: str) -> list[ChatMessage]:
    root = _require_study_root(sid)
    return [ChatMessage(**m) for m in mobile_chat.load_messages(root)]


async def _stream_chat(root: Path, sid: str, question: str, engine) -> AsyncIterator[bytes]:
    # 1. persist the user message immediately — reconnecting clients see it
    mobile_chat.append_message(root, role="user", content=question)

    # 2. load clips + build context from recent history
    try:
        study = _store().load_study(sid)
    except FileNotFoundError:
        yield _sse({"type": "error", "message": "Study missing"})
        return
    images, videos = collect_media(study.clips)

    prior = mobile_chat.load_messages(root)[:-1]  # exclude just-appended user msg
    query = _build_history_query(mobile_chat.context_window(prior), question)

    # 3. placeholder assistant row so a partial persist is addressable
    placeholder = mobile_chat.append_message(root, role="assistant", content="")
    placeholder_id = placeholder["id"]

    buffer: list[str] = []
    try:
        async for delta in engine.infer_stream(
            system=MOBILE_CHAT_SYSTEM,
            query=query,
            images=images,
            videos=videos,
        ):
            buffer.append(delta)
            yield _sse({"type": "token", "delta": delta})

    except asyncio.CancelledError:
        partial = "".join(buffer).strip()
        if partial:
            mobile_chat.replace_last_assistant(root, content=partial, message_id=placeholder_id)
        raise
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})
        return

    final_text = "".join(buffer).strip()
    if not final_text:
        yield _sse({"type": "error", "message": "Empty model response"})
        return

    saved = mobile_chat.replace_last_assistant(root, content=final_text, message_id=placeholder_id)
    yield _sse({"type": "done", "messageId": saved["id"]})


@router.post(
    "/studies/{sid}/chat",
    dependencies=[Depends(require_mobile_user)],
)
async def chat(sid: str, body: ChatBody, engine=Depends(get_engine)) -> StreamingResponse:
    root = _require_study_root(sid)  # 404 before we open the stream
    return StreamingResponse(
        _stream_chat(root, sid, body.content.strip(), engine),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
