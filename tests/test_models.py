from datetime import datetime
from app.models.study import Clip, Study, TasksAvailability


def test_clip_roundtrip():
    c = Clip(
        file_id="f1",
        original_filename="IM_1234",
        kind="dicom",
        raw_path="/x/raw/f1.dcm",
        converted_path="/x/conv/f1.mp4",
        view="Apical 4C 2D",
        user_view=None,
        confidence=0.87,
        is_video=True,
    )
    s = c.model_dump_json()
    c2 = Clip.model_validate_json(s)
    assert c2 == c


def test_tasks_availability_shape():
    ta = TasksAvailability(
        report=True, measurement=True, disease=True, vqa=True,
        missing_groups=["a2c", "ivc"],
    )
    assert ta.report is True
    assert "a2c" in ta.missing_groups


def test_study_roundtrip(tmp_path):
    s = Study(
        study_id="sid1",
        created_at=datetime(2026, 4, 18, 12, 0, 0),
        clips=[],
        tasks=TasksAvailability(report=True, measurement=True, disease=True,
                                vqa=True, missing_groups=[]),
    )
    s2 = Study.model_validate_json(s.model_dump_json())
    assert s2 == s


def test_task_schemas():
    from app.models.task import (
        TaskKind, TaskStatus, ReportResult, ReportSection,
        MeasurementItem, MeasurementResult,
        DiseaseItem, DiseaseResult,
        VQAMessage, VQAResult,
    )

    r = ReportResult(
        status="done",
        sections=[ReportSection(name="Left Ventricle", content="...", edited=False)],
    )
    assert r.sections[0].name == "Left Ventricle"

    m = MeasurementResult(
        status="done",
        items=[MeasurementItem(name="LV ejection fraction", value="55", unit="%", raw="...")],
    )
    assert m.items[0].unit == "%"

    d = DiseaseResult(
        status="done",
        items=[DiseaseItem(name="Aortic stenosis", answer="no", raw="No.")],
    )
    assert d.items[0].answer == "no"

    v = VQAResult(
        status="done",
        messages=[VQAMessage(role="user", content="What view?"),
                  VQAMessage(role="assistant", content="A4C.")],
    )
    assert v.messages[-1].role == "assistant"

    assert TaskKind.REPORT == "report"
    assert TaskStatus.RUNNING == "running"
