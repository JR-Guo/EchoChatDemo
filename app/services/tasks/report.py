from __future__ import annotations

import logging
import re
from typing import Sequence

from app.models.study import Clip
from app.models.task import ReportResult, ReportSection, TaskStatus
from app.services.progress import ProgressEvent, ProgressHub
from app.services.tasks.base import collect_media
from constants.prompts import REPORT_PROMPT
from constants.report_sections import REPORT_SECTIONS

_LOG = logging.getLogger("echochat.tasks.report")


def _split_sections(text: str) -> dict[str, str]:
    """Parse a model response by locating each known section header in order.

    The model often omits the trailing colon and writes everything on a few
    long lines, e.g.:
        "Aortic Valve The aortic valve is trileaflet. Atria The left atrial
         size is normal. Great Vessels ..."

    We split by searching for every section name (whole-word, case-sensitive
    since the trained model reproduces them with the correct capitalisation)
    and take whatever text sits between one header and the next.
    """
    out: dict[str, str] = {s: "" for s in REPORT_SECTIONS}

    if not text:
        return out

    header_re = re.compile(
        r"(?:^|[^A-Za-z])("
        + "|".join(re.escape(s) for s in REPORT_SECTIONS)
        + r")(?=[^A-Za-z]|$)"
    )

    hits: list[tuple[str, int, int]] = []  # (name, start, end-after-name)
    for m in header_re.finditer(text):
        name = m.group(1)
        hits.append((name, m.start(1), m.end(1)))

    # Fallback: case-insensitive search if strict-case missed everything
    if not hits:
        ci_re = re.compile(
            r"(?:^|[^A-Za-z])("
            + "|".join(re.escape(s) for s in REPORT_SECTIONS)
            + r")(?=[^A-Za-z]|$)",
            re.IGNORECASE,
        )
        for m in ci_re.finditer(text):
            # Map back to canonical capitalisation
            canonical = next(
                s for s in REPORT_SECTIONS if s.lower() == m.group(1).lower()
            )
            hits.append((canonical, m.start(1), m.end(1)))

    if not hits:
        return out

    # Extract content between each header and the next.
    for i, (name, _start, end) in enumerate(hits):
        next_start = hits[i + 1][1] if i + 1 < len(hits) else len(text)
        body = text[end:next_start]
        # Strip a single leading separator (colon / dash / em-dash / space)
        body = re.sub(r"^[\s:\-\u2014]+", "", body)
        body = body.strip()
        if body:
            # A section may legitimately appear twice in the output; keep the
            # first non-empty occurrence.
            if not out[name]:
                out[name] = body

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

    _LOG.info("task=%s raw output (%d chars): %s", task_id, len(raw), raw)

    sections_map = _split_sections(raw)

    if all(not v for v in sections_map.values()) and raw.strip():
        sections_map["Summary"] = raw.strip()

    sections = [ReportSection(name=name, content=sections_map.get(name, "").strip())
                for name in REPORT_SECTIONS]

    for s in sections:
        await hub.publish(task_id, ProgressEvent(kind="partial",
                         data={"section": s.name, "content": s.content}))

    result = ReportResult(status=TaskStatus.DONE, sections=sections)
    await hub.publish(task_id, ProgressEvent(kind="done", data={"task_id": task_id}))
    return result
