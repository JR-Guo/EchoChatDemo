from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.mobile_auth import require_mobile_user, sign_path
from app.models.study import Study
from app.storage import Storage


router = APIRouter(tags=["mobile-studies"], dependencies=[Depends(require_mobile_user)])


StudyStatus = Literal["queued", "processing", "ready", "failed"]


class StudySummary(BaseModel):
    id: str
    title: str = ""
    patientId: str = ""
    status: StudyStatus
    createdAt: str
    thumbnailUrl: Optional[str] = None
    viewCount: int


class ViewInfo(BaseModel):
    name: str
    confidence: float = 0.0
    thumbnailUrl: Optional[str] = None
    mp4Url: Optional[str] = None


class StudyDetail(StudySummary):
    views: list[ViewInfo] = Field(default_factory=list)
    reportId: Optional[str] = None


class StudiesPage(BaseModel):
    items: list[StudySummary]
    nextCursor: Optional[str] = None


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


def _encode_cursor(skip: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"skip": skip}).encode()).decode()


def _decode_cursor(cursor: str) -> int:
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        skip = int(data["skip"])
        if skip < 0:
            raise ValueError
        return skip
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Bad cursor")


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_soft_deleted(study_root: Path) -> bool:
    return (study_root / ".deleted").exists()


def _load_all_studies() -> list[Study]:
    store = _store()
    sessions = store.base / "sessions"
    if not sessions.exists():
        return []
    studies: list[Study] = []
    for entry in sessions.iterdir():
        if not entry.is_dir() or _is_soft_deleted(entry):
            continue
        meta = entry / "meta.json"
        if not meta.exists():
            continue
        try:
            studies.append(Study.model_validate_json(meta.read_text()))
        except Exception:
            # Corrupt or in-progress study — skip rather than 500 the whole list
            continue
    studies.sort(key=lambda s: s.created_at, reverse=True)
    return studies


def _status_for(study: Study) -> StudyStatus:
    """Loose M1 derivation: ready if any clip has a view, else processing/queued."""
    if not study.clips:
        return "queued"
    if any(c.effective_view for c in study.clips):
        return "ready"
    return "processing"


def _thumbnail_url_for(study: Study) -> Optional[str]:
    if not study.clips:
        return None
    first = study.clips[0]
    return sign_path(f"/api/mobile/v1/studies/{study.study_id}/clips/{first.file_id}/thumbnail")


def _summarize(study: Study) -> StudySummary:
    return StudySummary(
        id=study.study_id,
        title="",
        patientId="",
        status=_status_for(study),
        createdAt=_iso_utc(study.created_at),
        thumbnailUrl=_thumbnail_url_for(study),
        viewCount=len(study.clips),
    )


def _report_id_for(study: Study) -> Optional[str]:
    store = _store()
    exports = store.study_root(study.study_id) / "exports"
    if (exports / "report.pdf").exists() or (exports / "report.docx").exists():
        return f"{study.study_id}-report"
    return None


def _views_for(study: Study) -> list[ViewInfo]:
    views: list[ViewInfo] = []
    for clip in study.clips:
        view_name = clip.effective_view
        if not view_name:
            continue
        thumb = sign_path(f"/api/mobile/v1/studies/{study.study_id}/clips/{clip.file_id}/thumbnail")
        mp4 = None
        if clip.converted_path and clip.is_video:
            mp4 = sign_path(f"/api/mobile/v1/studies/{study.study_id}/clips/{clip.file_id}/video")
        views.append(ViewInfo(
            name=view_name,
            confidence=clip.confidence or 0.0,
            thumbnailUrl=thumb,
            mp4Url=mp4,
        ))
    return views


@router.get("/studies", response_model=StudiesPage)
def list_studies(
    cursor: Optional[str] = None,
    limit: int = Query(20, ge=1, le=50),
) -> StudiesPage:
    skip = _decode_cursor(cursor) if cursor else 0
    studies = _load_all_studies()
    page = studies[skip : skip + limit]
    next_cursor = _encode_cursor(skip + limit) if (skip + limit) < len(studies) else None
    return StudiesPage(
        items=[_summarize(s) for s in page],
        nextCursor=next_cursor,
    )


@router.get("/studies/{sid}", response_model=StudyDetail)
def get_study(sid: str) -> StudyDetail:
    store = _store()
    root = store.study_root(sid)
    if not root.exists() or _is_soft_deleted(root):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Study not found")
    try:
        study = store.load_study(sid)
    except FileNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Study not found")

    summary = _summarize(study)
    return StudyDetail(
        **summary.model_dump(),
        views=_views_for(study),
        reportId=_report_id_for(study),
    )


@router.delete("/studies/{sid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_study(sid: str) -> Response:
    store = _store()
    root = store.study_root(sid)
    if not root.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Study not found")
    (root / ".deleted").write_text(datetime.now(tz=timezone.utc).isoformat())
    return Response(status_code=status.HTTP_204_NO_CONTENT)
