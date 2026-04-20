from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Response
from fastapi.responses import FileResponse

from app.config import get_settings
from app.storage import Storage
from app.models.study import TasksAvailability
from constants.view_labels import VIEW_LABELS, view_coarse_group


router = APIRouter()


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


def _recompute_availability(study) -> TasksAvailability:
    if not study.clips:
        return TasksAvailability(report=False, measurement=False, disease=False,
                                 vqa=False, missing_groups=["any"])

    seen_groups: set[str] = set()
    for c in study.clips:
        v = c.effective_view
        if v:
            seen_groups.add(view_coarse_group(v))

    baseline = {"a4c", "plax", "psax"}
    missing = sorted(baseline - seen_groups)

    return TasksAvailability(
        report=True, measurement=True, disease=True, vqa=True, missing_groups=missing,
    )


@router.get("/api/study/{sid}")
def get_study(sid: str):
    s = _store()
    study = s.load_study(sid)
    study.tasks = _recompute_availability(study)
    s.save_study(study)
    return study.model_dump()


@router.patch("/api/study/{sid}/clip/{fid}")
def patch_clip(sid: str, fid: str, body: dict = Body(...)):
    if "user_view" not in body:
        raise HTTPException(status_code=400, detail="user_view required")
    uv = body["user_view"]
    if uv is not None and uv not in VIEW_LABELS:
        raise HTTPException(status_code=400, detail=f"user_view must be in VIEW_LABELS or null")

    s = _store()
    study = s.load_study(sid)
    for clip in study.clips:
        if clip.file_id == fid:
            clip.user_view = uv
            study.tasks = _recompute_availability(study)
            s.save_study(study)
            return clip.model_dump()
    raise HTTPException(status_code=404, detail="clip not found")


@router.delete("/api/study/{sid}/clip/{fid}", status_code=204)
def delete_clip(sid: str, fid: str):
    s = _store()
    study = s.load_study(sid)
    before = len(study.clips)
    study.clips = [c for c in study.clips if c.file_id != fid]
    if len(study.clips) == before:
        raise HTTPException(status_code=404, detail="clip not found")
    # Move related files into a per-study 'trash' folder instead of deleting.
    trash = Path(s.study_root(sid)) / "trash"
    trash.mkdir(parents=True, exist_ok=True)
    for pat in ("raw/*", "converted/*"):
        for p in Path(s.study_root(sid)).glob(pat):
            if fid in p.name:
                shutil.move(str(p), str(trash / p.name))
    study.tasks = _recompute_availability(study)
    s.save_study(study)
    return Response(status_code=204)


@router.get("/api/study/{sid}/clip/{fid}/thumbnail")
def thumbnail(sid: str, fid: str):
    s = _store()
    p = s.thumbnail_path(sid, fid)
    if not p.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(p), media_type="image/png")


@router.get("/api/study/{sid}/clip/{fid}/video")
def clip_video(sid: str, fid: str):
    s = _store()
    study = s.load_study(sid)
    match = next((c for c in study.clips if c.file_id == fid), None)
    if not match or not match.converted_path:
        raise HTTPException(status_code=404)
    p = Path(match.converted_path)
    if not p.exists():
        raise HTTPException(status_code=404)
    mt = "video/mp4" if match.is_video else "image/png"
    return FileResponse(str(p), media_type=mt)
