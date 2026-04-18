import asyncio

from app.services.tasks.disease import run_disease
from app.services.progress import ProgressHub
from app.models.study import Clip


class SeqEngine:
    def __init__(self, answers): self.answers = list(answers)
    async def infer(self, system, query, images, videos):
        return self.answers.pop(0)


async def test_disease_parses_yes_no_and_unknown():
    hub = ProgressHub()
    engine = SeqEngine([
        "Yes, there is aortic stenosis.",
        "No, not present.",
        "I cannot determine based on the images.",
    ])
    clips = [Clip(file_id="f1", original_filename="x", kind="dicom",
                  raw_path="/x", converted_path="/a.mp4", is_video=True)]
    events: list[dict] = []
    async def reader():
        async for e in hub.subscribe("d1"):
            events.append({"kind": e.kind, **e.data})
            if e.kind in ("done", "error"):
                break
    r = asyncio.create_task(reader())
    result = await run_disease(
        task_id="d1", clips=clips,
        items=["Aortic stenosis", "Mitral regurgitation", "Hypertrophic cardiomyopathy"],
        engine=engine, hub=hub,
    )
    await asyncio.wait_for(r, timeout=1.0)

    answers = [i.answer for i in result.items]
    assert answers == ["yes", "no", "unknown"]
    assert len([e for e in events if e["kind"] == "item"]) == 3
