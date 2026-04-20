import pytest

from app.services.tasks.report import run_report
from app.services.progress import ProgressHub
from app.models.study import Clip


class FakeEngine:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def infer(self, system, query, images, videos):
        self.calls += 1
        return self.text


async def test_report_emits_sections(tmp_path):
    hub = ProgressHub()
    engine = FakeEngine(
        "Aortic Valve: normal.\n"
        "Atria: normal.\n"
        "Great Vessels: normal.\n"
        "Left Ventricle: EF 55.\n"
        "Mitral Valve: normal.\n"
        "Pericardium Pleural: no effusion.\n"
        "Pulmonic Valve: normal.\n"
        "Right Ventricle: normal.\n"
        "Tricuspid Valve: trivial TR.\n"
        "Summary: unremarkable study."
    )
    clips = [Clip(file_id="f1", original_filename="x", kind="dicom",
                  raw_path="/x", converted_path="/a.mp4", is_video=True)]

    events: list[dict] = []
    task_id = "tid1"

    async def reader():
        async for evt in hub.subscribe(task_id):
            events.append({"kind": evt.kind, **evt.data})
            if evt.kind in ("done", "error"):
                break

    import asyncio
    r = asyncio.create_task(reader())
    result = await run_report(task_id=task_id, clips=clips, engine=engine, hub=hub)
    await asyncio.wait_for(r, timeout=1.0)

    assert result.status == "done"
    assert len(result.sections) == 10
    assert result.sections[0].name == "Aortic Valve"
    assert "normal" in result.sections[0].content.lower()

    assert any(e.get("section") == "Left Ventricle" for e in events if e["kind"] == "partial")
    assert any(e["kind"] == "done" for e in events)


async def test_report_missing_section_preserves_placeholder(tmp_path):
    hub = ProgressHub()
    engine = FakeEngine("Summary: minimal response only.")
    clips = [Clip(file_id="f1", original_filename="x", kind="dicom",
                  raw_path="/x", converted_path="/a.mp4", is_video=True)]

    result = await run_report(task_id="t2", clips=clips, engine=engine, hub=hub)
    names = [s.name for s in result.sections]
    assert names.count("Summary") == 1
    summary = next(s for s in result.sections if s.name == "Summary")
    assert "minimal" in summary.content.lower()


async def test_report_parses_model_output_without_colons(tmp_path):
    hub = ProgressHub()
    engine = FakeEngine(
        "Aortic Valve The aortic valve is trileaflet. "
        "Atria The left atrial size is normal. "
        "Great Vessels The aortic root is normal size. "
        "Left Ventricle EF 60%. "
        "Mitral Valve Normal. "
        "Pericardium Pleural No effusion. "
        "Pulmonic Valve Not well visualized. "
        "Right Ventricle Normal. "
        "Tricuspid Valve Trace TR. "
        "Summary Unremarkable."
    )
    clips = [Clip(file_id="f1", original_filename="x", kind="dicom",
                  raw_path="/x", converted_path="/a.mp4", is_video=True)]
    result = await run_report(task_id="t3", clips=clips, engine=engine, hub=hub)
    assert result.status == "done"
    by_name = {s.name: s.content for s in result.sections}
    assert "trileaflet" in by_name["Aortic Valve"]
    assert "left atrial" in by_name["Atria"]
    assert "aortic root" in by_name["Great Vessels"]
    assert "EF 60%" in by_name["Left Ventricle"]
    assert "Unremarkable" in by_name["Summary"]
