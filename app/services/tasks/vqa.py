from __future__ import annotations

from typing import Sequence

from app.models.study import Clip
from app.models.task import TaskStatus, VQAMessage, VQAResult
from app.services.progress import ProgressEvent, ProgressHub
from app.services.tasks.base import collect_media
from constants.prompts import VQA_PROMPT


async def run_vqa(
    *, task_id: str, clips: Sequence[Clip], question: str, engine, hub: ProgressHub,
) -> VQAResult:
    if not question.strip():
        raise ValueError("question is empty")
    images, videos = collect_media(clips)

    await hub.publish(task_id, ProgressEvent(kind="phase", data={"phase": "thinking"}))
    try:
        raw = await engine.infer(
            system=VQA_PROMPT.system,
            query=VQA_PROMPT.query_template.format(question=question),
            images=images, videos=videos,
        )
    except Exception as e:
        await hub.publish(task_id, ProgressEvent(kind="error", data={"reason": str(e)}))
        return VQAResult(status=TaskStatus.ERROR, error=str(e))

    msgs = [
        VQAMessage(role="user", content=question.strip()),
        VQAMessage(role="assistant", content=raw.strip()),
    ]
    await hub.publish(task_id, ProgressEvent(kind="message", data=msgs[-1].model_dump()))
    await hub.publish(task_id, ProgressEvent(kind="done", data={"task_id": task_id}))
    return VQAResult(status=TaskStatus.DONE, messages=msgs)
