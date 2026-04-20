from __future__ import annotations

import re
from typing import Sequence

from app.models.study import Clip
from app.models.task import DiseaseItem, DiseaseResult, TaskStatus
from app.services.progress import ProgressEvent, ProgressHub
from app.services.tasks.base import collect_media
from constants.diseases import SUPPORT_DISEASES
from constants.prompts import DISEASE_PROMPT


_YES = re.compile(r"^\s*(yes|yeah|present|affirmative)\b", re.IGNORECASE)
_NO = re.compile(r"^\s*(no|not present|negative|absent)\b", re.IGNORECASE)


def _parse_yn(text: str) -> str:
    if _YES.search(text):
        return "yes"
    if _NO.search(text):
        return "no"
    first = text.strip().split("\n", 1)[0]
    if _YES.search(first):
        return "yes"
    if _NO.search(first):
        return "no"
    return "unknown"


async def run_disease(
    *, task_id: str, clips: Sequence[Clip], items: Sequence[str], engine, hub: ProgressHub,
) -> DiseaseResult:
    for it in items:
        if it not in SUPPORT_DISEASES:
            raise ValueError(f"unsupported disease: {it}")

    images, videos = collect_media(clips)
    out: list[DiseaseItem] = []
    for i, name in enumerate(items, start=1):
        await hub.publish(task_id, ProgressEvent(kind="phase",
                         data={"phase": "during", "i": i, "n": len(items), "name": name}))
        query = DISEASE_PROMPT.query_template.replace("<disease>", name)
        try:
            raw = await engine.infer(
                system=DISEASE_PROMPT.system, query=query,
                images=images, videos=videos,
            )
        except Exception as e:
            await hub.publish(task_id, ProgressEvent(kind="error",
                             data={"reason": str(e), "name": name}))
            return DiseaseResult(status=TaskStatus.ERROR, items=out, error=str(e))

        item = DiseaseItem(name=name, answer=_parse_yn(raw), raw=raw.strip())
        out.append(item)
        await hub.publish(task_id, ProgressEvent(kind="item", data=item.model_dump()))

    await hub.publish(task_id, ProgressEvent(kind="done", data={"task_id": task_id}))
    return DiseaseResult(status=TaskStatus.DONE, items=out)
