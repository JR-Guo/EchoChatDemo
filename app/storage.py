from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.study import Study


class Storage:
    """Filesystem accessor for /<base>/sessions/<study_id>/ layout."""

    def __init__(self, base: Path):
        self.base = Path(base)
        (self.base / "sessions").mkdir(parents=True, exist_ok=True)

    # --- ids ---
    def new_study_id(self) -> str:
        return uuid.uuid4().hex[:16]

    # --- paths ---
    def study_root(self, study_id: str) -> Path:
        return self.base / "sessions" / study_id

    def ensure_study(self, study_id: str) -> Path:
        root = self.study_root(study_id)
        for sub in ("raw", "converted", "results", "exports"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        meta = root / "meta.json"
        if not meta.exists():
            initial = Study(
                study_id=study_id,
                created_at=datetime.utcnow(),
                clips=[],
                tasks=_default_availability(),
            )
            meta.write_text(initial.model_dump_json(indent=2))
        return root

    def raw_path(self, study_id: str, file_id: str, suffix: str) -> Path:
        return self.study_root(study_id) / "raw" / f"{file_id}{suffix}"

    def converted_path(self, study_id: str, file_id: str, suffix: str) -> Path:
        return self.study_root(study_id) / "converted" / f"{file_id}{suffix}"

    def thumbnail_path(self, study_id: str, file_id: str) -> Path:
        return self.study_root(study_id) / "converted" / f"{file_id}.thumb.png"

    def result_path(self, study_id: str, filename: str) -> Path:
        return self.study_root(study_id) / "results" / filename

    def export_path(self, study_id: str, filename: str) -> Path:
        return self.study_root(study_id) / "exports" / filename

    # --- meta ---
    def load_study(self, study_id: str) -> Study:
        meta = self.study_root(study_id) / "meta.json"
        return Study.model_validate_json(meta.read_text())

    def save_study(self, study: Study) -> None:
        self.ensure_study(study.study_id)
        meta = self.study_root(study.study_id) / "meta.json"
        meta.write_text(study.model_dump_json(indent=2))

    def save_result(self, study_id: str, filename: str, data: Any) -> None:
        p = self.result_path(study_id, filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _default_availability():
    from app.models.study import TasksAvailability
    return TasksAvailability(
        report=False, measurement=False, disease=False, vqa=False, missing_groups=[],
    )
