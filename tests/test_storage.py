import json
from datetime import datetime
from pathlib import Path

import pytest

from app.models.study import Clip, Study, TasksAvailability
from app.storage import Storage


@pytest.fixture
def store(tmp_path):
    return Storage(base=tmp_path)


def test_new_study_creates_directories(store, tmp_path):
    sid = store.new_study_id()
    store.ensure_study(sid)
    root = tmp_path / "sessions" / sid
    assert (root / "raw").is_dir()
    assert (root / "converted").is_dir()
    assert (root / "results").is_dir()
    assert (root / "exports").is_dir()


def test_save_and_load_study(store):
    sid = store.new_study_id()
    store.ensure_study(sid)
    s = Study(
        study_id=sid,
        created_at=datetime(2026, 4, 18, 12),
        clips=[Clip(file_id="f1", original_filename="x.dcm", kind="dicom",
                    raw_path="/tmp/x.dcm")],
        tasks=TasksAvailability(report=True, measurement=True, disease=True,
                                vqa=True, missing_groups=[]),
    )
    store.save_study(s)
    loaded = store.load_study(sid)
    assert loaded == s


def test_raw_converted_paths(store):
    sid = store.new_study_id()
    store.ensure_study(sid)
    p = store.raw_path(sid, "f1", ".dcm")
    assert str(p).endswith("raw/f1.dcm")
    q = store.converted_path(sid, "f1", ".mp4")
    assert str(q).endswith("converted/f1.mp4")


def test_save_result_json(store):
    sid = store.new_study_id()
    store.ensure_study(sid)
    store.save_result(sid, "report.json", {"sections": [{"name": "LV", "content": "ok"}]})
    data = json.loads(Path(store.result_path(sid, "report.json")).read_text())
    assert data["sections"][0]["name"] == "LV"
