from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException

from app.config import get_settings
from app.storage import Storage
from constants.report_sections import REPORT_SECTIONS

router = APIRouter()


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


@router.patch("/api/study/{sid}/report/section")
def patch_section(sid: str, body: dict = Body(...)):
    s = _store()
    name = body.get("section")
    content = body.get("content", "")
    if name not in REPORT_SECTIONS:
        raise HTTPException(status_code=400, detail="invalid section name")
    p = s.result_path(sid, "report.json")
    if not p.exists():
        raise HTTPException(status_code=404, detail="report not generated yet")
    data = json.loads(Path(p).read_text())
    found = False
    for sec in data.get("sections", []):
        if sec["name"] == name:
            sec["content"] = content
            sec["edited"] = True
            found = True
            break
    if not found:
        data.setdefault("sections", []).append(
            {"name": name, "content": content, "edited": True}
        )
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return {"name": name, "content": content, "edited": True}
