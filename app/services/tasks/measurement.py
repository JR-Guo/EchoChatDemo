from __future__ import annotations

from typing import Sequence

from app.models.study import Clip
from app.models.task import MeasurementItem, MeasurementResult, TaskStatus
from app.services.progress import ProgressEvent, ProgressHub
from app.services.tasks.base import collect_media, split_value_unit
from constants.measurements import SUPPORT_MEASUREMENTS
from constants.prompts import MEASUREMENT_PROMPT


async def run_measurement(
    *, task_id: str, clips: Sequence[Clip], items: Sequence[str], engine, hub: ProgressHub,
) -> MeasurementResult:
    for it in items:
        if it not in SUPPORT_MEASUREMENTS:
            raise ValueError(f"unsupported measurement: {it}")

    images, videos = collect_media(clips)
    out: list[MeasurementItem] = []
    total = len(items)
    for i, name in enumerate(items, start=1):
        await hub.publish(task_id, ProgressEvent(kind="phase",
                         data={"phase": "during", "i": i, "n": total, "name": name}))
        query = MEASUREMENT_PROMPT.query_template.replace("<measure>", name)
        try:
            raw = await engine.infer(
                system=MEASUREMENT_PROMPT.system, query=query,
                images=images, videos=videos,
            )
        except Exception as e:
            await hub.publish(task_id, ProgressEvent(kind="error",
                             data={"reason": str(e), "name": name}))
            return MeasurementResult(status=TaskStatus.ERROR, items=out, error=str(e))

        value, unit = split_value_unit(raw)
        item = MeasurementItem(name=name, value=value, unit=unit, raw=raw.strip())
        out.append(item)
        await hub.publish(task_id, ProgressEvent(kind="item", data=item.model_dump()))

    await hub.publish(task_id, ProgressEvent(kind="done", data={"task_id": task_id}))
    return MeasurementResult(status=TaskStatus.DONE, items=out)
