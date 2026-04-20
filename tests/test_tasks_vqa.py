import asyncio

from app.services.tasks.vqa import run_vqa
from app.services.progress import ProgressHub
from app.models.study import Clip


class OneShotEngine:
    def __init__(self, resp): self.resp = resp
    async def infer(self, system, query, images, videos): return self.resp


async def test_vqa_returns_assistant_message():
    hub = ProgressHub()
    clips = [Clip(file_id="f1", original_filename="x", kind="dicom",
                  raw_path="/x", converted_path="/a.mp4", is_video=True)]
    events: list[dict] = []
    async def reader():
        async for e in hub.subscribe("v1"):
            events.append({"kind": e.kind, **e.data})
            if e.kind in ("done", "error"):
                break
    r = asyncio.create_task(reader())
    result = await run_vqa(
        task_id="v1", clips=clips,
        question="What view is shown?",
        engine=OneShotEngine("It is A4C."),
        hub=hub,
    )
    await asyncio.wait_for(r, timeout=1.0)

    assert result.status == "done"
    assert result.messages[-1].role == "assistant"
    assert "A4C" in result.messages[-1].content
    assert any(e["kind"] == "message" for e in events)
