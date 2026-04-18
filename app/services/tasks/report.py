from __future__ import annotations

import re
from typing import Sequence

from app.models.study import Clip
from app.models.task import ReportResult, ReportSection, TaskStatus
from app.services.progress import ProgressEvent, ProgressHub
from app.services.tasks.base import collect_media
from constants.prompts import REPORT_PROMPT
from constants.report_sections import REPORT_SECTIONS


def _split_sections(text: str) -> dict[str, str]:
    """Parse a model response with 'Section Name: content' lines."""
    out: dict[str, str] = {s: "" for s in REPORT_SECTIONS}
    pattern = re.compile(
        r"^\s*(" + "|".join(re.escape(s) for s in REPORT_SECTIONS) + r")\s*[:\-\u2014]\s*(.*)$",
        re.IGNORECASE,
    )

    current: str | None = None
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            name = next(s for s in REPORT_SECTIONS if s.lower() == m.group(1).lower())
            current = name
            out[current] = (out[current] + ("\n" if out[current] else "") + m.group(2)).strip()
        elif current and line.strip():
            out[current] = (out[current] + "\n" + line).strip()
    return out


async def run_report(
    *, task_id: str, clips: Sequence[Clip], engine, hub: ProgressHub
) -> ReportResult:
    images, videos = collect_media(clips)

    await hub.publish(task_id, ProgressEvent(kind="phase",
                     data={"phase": "preparing_context"}))
    try:
        await hub.publish(task_id, ProgressEvent(kind="phase",
                         data={"phase": "inference"}))
        raw = await engine.infer(
            system=REPORT_PROMPT.system,
            query=REPORT_PROMPT.query_template,
            images=images,
            videos=videos,
        )
    except Exception as e:
        res = ReportResult(status=TaskStatus.ERROR, error=str(e))
        await hub.publish(task_id, ProgressEvent(kind="error",
                         data={"reason": str(e)}))
        return res

    sections_map = _split_sections(raw)
    sections = [ReportSection(name=name, content=sections_map.get(name, "").strip())
                for name in REPORT_SECTIONS]

    for s in sections:
        await hub.publish(task_id, ProgressEvent(kind="partial",
                         data={"section": s.name, "content": s.content}))

    result = ReportResult(status=TaskStatus.DONE, sections=sections)
    await hub.publish(task_id, ProgressEvent(kind="done", data={"task_id": task_id}))
    return result
