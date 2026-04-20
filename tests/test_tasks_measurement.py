import asyncio

from app.services.tasks.measurement import run_measurement
from app.services.progress import ProgressHub
from app.models.study import Clip


class SeqEngine:
    def __init__(self, answers):
        self.answers = list(answers)
    async def infer(self, system, query, images, videos):
        return self.answers.pop(0)


async def test_measurement_emits_items_and_persists_order():
    hub = ProgressHub()
    engine = SeqEngine(["55 %", "38 mm"])
    clips = [Clip(file_id="f1", original_filename="x", kind="dicom",
                  raw_path="/x", converted_path="/a.mp4", is_video=True)]

    events: list[dict] = []
    async def reader():
        async for evt in hub.subscribe("m1"):
            events.append({"kind": evt.kind, **evt.data})
            if evt.kind in ("done", "error"):
                break
    r = asyncio.create_task(reader())
    result = await run_measurement(
        task_id="m1",
        clips=clips,
        items=["LV ejection fraction", "Left-atrial antero-posterior dimension"],
        engine=engine,
        hub=hub,
    )
    await asyncio.wait_for(r, timeout=1.0)

    assert result.status == "done"
    names = [i.name for i in result.items]
    assert names == ["LV ejection fraction", "Left-atrial antero-posterior dimension"]
    assert result.items[0].value == "55"
    assert result.items[0].unit == "%"
    assert result.items[1].unit == "mm"
    item_events = [e for e in events if e["kind"] == "item"]
    assert len(item_events) == 2


async def test_measurement_rejects_unknown_item():
    hub = ProgressHub()
    engine = SeqEngine([])
    import pytest
    with pytest.raises(ValueError):
        await run_measurement(task_id="m2", clips=[], items=["Bogus"], engine=engine, hub=hub)
