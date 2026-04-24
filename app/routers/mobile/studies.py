from __future__ import annotations

import base64
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.mobile_auth import require_mobile_user, sign_path
from app.models.study import Clip, Study
from app.services import mobile_media
from app.services.mobile_uploads import UploadStore
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


class CreateStudyBody(BaseModel):
    uploadId: str = Field(min_length=1, max_length=64)
    title: Optional[str] = Field(default=None, max_length=200)


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


def _upload_store() -> UploadStore:
    return UploadStore(base_dir=Path(get_settings().data_dir) / "uploads")


def _build_clip(*, file_id: str, original_filename: str, kind: str, raw_path: Path,
                converted_path: Optional[Path], is_video: bool) -> Clip:
    return Clip(
        file_id=file_id,
        original_filename=original_filename,
        kind=kind,  # type: ignore[arg-type]
        raw_path=str(raw_path),
        converted_path=str(converted_path) if converted_path else None,
        is_video=is_video,
    )


def _install_non_dicom(*, kind: str, upload_data: Path, original_filename: str,
                      converted_dir: Path) -> Clip:
    if kind == "video":
        file_id, dest, _thumb = mobile_media.install_video(
            upload_data=upload_data,
            original_filename=original_filename,
            converted_dir=converted_dir,
        )
        is_video = True
    else:  # image
        file_id, dest, _thumb = mobile_media.install_image(
            upload_data=upload_data,
            original_filename=original_filename,
            converted_dir=converted_dir,
        )
        is_video = False
    return _build_clip(
        file_id=file_id,
        original_filename=original_filename,
        kind=kind,
        raw_path=str(dest),
        converted_path=dest,
        is_video=is_video,
    )


def _install_dicom(*, upload_data: Path, original_filename: str,
                   study_root: Path) -> list[Clip]:
    """Copy the uploaded DICOM into `raw/`, run the existing preprocess
    pipeline, and return one Clip per selected media output.

    Heavy: invokes view classification + DICOM-to-mp4 conversion. Typical
    latency ~20-60s per file. Synchronous on purpose — matches the existing
    web upload behavior. Gunicorn timeout is 600s in prod.
    """
    from app.services.preprocess_adapter import run_study_preprocess

    file_id = mobile_media.new_file_id()
    raw_dir = study_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_dest = raw_dir / f"{file_id}.dcm"
    mobile_media._move_or_copy(upload_data, raw_dest)

    try:
        result = run_study_preprocess(
            raw_dir=raw_dir,
            all_output_dir=study_root / "preprocessed" / "all",
            selected_output_dir=study_root / "preprocessed" / "selected",
            clip_ids=[file_id],
        )
    except Exception as exc:
        # Leave raw file on disk; surface the error so client can retry
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DICOM preprocess failed: {exc}",
        )

    clips: list[Clip] = []
    media_map = result.selected or result.all_media
    if not media_map:
        # Preprocess produced nothing usable — keep the raw clip anyway so
        # the user sees it appeared on the server and can retry/delete.
        clips.append(_build_clip(
            file_id=file_id,
            original_filename=original_filename,
            kind="dicom",
            raw_path=raw_dest,
            converted_path=None,
            is_video=True,
        ))
        return clips

    for fid, media in media_map.items():
        clips.append(_build_clip(
            file_id=fid,
            original_filename=original_filename,
            kind="dicom",
            raw_path=raw_dest,
            converted_path=media.path,
            is_video=media.is_video,
        ))
    return clips


@router.post(
    "/studies",
    response_model=StudyDetail,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_mobile_user)],
)
def create_study_from_upload(body: CreateStudyBody) -> StudyDetail:
    uploads = _upload_store()
    meta = uploads.consume(body.uploadId)
    if meta is None:
        # Either missing, or still in progress
        raw_meta = uploads.load(body.uploadId)
        if raw_meta is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Upload not found")
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Upload is not complete")

    data_path = uploads.data_path(body.uploadId)
    kind = mobile_media.detect_kind(data_path, filename_hint=meta.filename)
    if kind is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported file type (accepted: DICOM, mp4, mov, png, jpg)",
        )

    store = _store()
    sid = store.new_study_id()
    study_root = store.ensure_study(sid)

    try:
        if kind == "dicom":
            clips = _install_dicom(
                upload_data=data_path,
                original_filename=meta.filename,
                study_root=study_root,
            )
        else:
            clips = [_install_non_dicom(
                kind=kind,
                upload_data=data_path,
                original_filename=meta.filename,
                converted_dir=study_root / "converted",
            )]
    except HTTPException:
        # clean up empty study so we don't leak dirs on a hard failure
        shutil.rmtree(study_root, ignore_errors=True)
        raise
    finally:
        uploads.delete(body.uploadId)

    study = store.load_study(sid)
    study.clips = clips
    if body.title:
        # Title isn't in the base Study model; keep it in the Clip's original_filename
        # for display (mobile client prefers title when present — extend later).
        pass
    store.save_study(study)

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
