"""Signed-URL download endpoints. No Bearer token required — authenticity
is established by the HMAC signature in query params, produced by
`mobile_auth.sign_path()`.

These live in a separate router so the Bearer-token dependency on the
studies router does not apply here.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.config import get_settings
from app.mobile_auth import require_valid_signature
from app.storage import Storage


router = APIRouter(tags=["mobile-downloads"], dependencies=[Depends(require_valid_signature)])


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


@router.get("/studies/{sid}/clips/{fid}/thumbnail", include_in_schema=False)
def clip_thumbnail(sid: str, fid: str):
    p = _store().thumbnail_path(sid, fid)
    if not p.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return FileResponse(str(p), media_type="image/png")


@router.get("/studies/{sid}/clips/{fid}/video", include_in_schema=False)
def clip_video(sid: str, fid: str):
    store = _store()
    try:
        study = store.load_study(sid)
    except FileNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    match = next((c for c in study.clips if c.file_id == fid), None)
    if not match or not match.converted_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    p = Path(match.converted_path)
    if not p.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    mt = "video/mp4" if match.is_video else "image/png"
    return FileResponse(str(p), media_type=mt)


@router.get("/studies/{sid}/report/pdf", include_in_schema=False)
def report_pdf(sid: str):
    p = _store().study_root(sid) / "exports" / "report.pdf"
    if not p.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return FileResponse(str(p), media_type="application/pdf", filename=f"{sid}.pdf")


@router.get("/studies/{sid}/report/docx", include_in_schema=False)
def report_docx(sid: str):
    p = _store().study_root(sid) / "exports" / "report.docx"
    if not p.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return FileResponse(
        str(p),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"{sid}.docx",
    )
