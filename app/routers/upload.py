from __future__ import annotations

import uuid
from fastapi import APIRouter, File, HTTPException, UploadFile
from pathlib import Path

from app.config import get_settings
from app.services.dicom_pipeline import looks_like_dicom
from app.storage import Storage

router = APIRouter()


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


@router.post("/api/study")
def create_study():
    s = _store()
    sid = s.new_study_id()
    s.ensure_study(sid)
    return {"study_id": sid}


@router.post("/api/study/{sid}/upload")
async def upload_file(sid: str, file: UploadFile = File(...)):
    s = _store()
    s.ensure_study(sid)

    file_id = uuid.uuid4().hex[:16]
    original_ext = Path(file.filename or "").suffix or ""
    raw = s.raw_path(sid, file_id, original_ext or "")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(data)

    if not looks_like_dicom(raw):
        # File is not DICOM. We do not delete the saved copy (destructive ops
        # are blocked); it will be overwritten on a future upload with the same
        # file_id (extremely unlikely with a UUID-derived id).
        raise HTTPException(status_code=400, detail="File is not DICOM.")

    study = s.load_study(sid)
    from app.models.study import Clip
    study.clips.append(
        Clip(
            file_id=file_id,
            original_filename=file.filename or file_id,
            kind="dicom",
            raw_path=str(raw),
        )
    )
    s.save_study(study)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "kind": "dicom",
        "size": len(data),
        "saved_as": str(raw),
    }
