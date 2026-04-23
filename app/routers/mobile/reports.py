from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.config import get_settings
from app.mobile_auth import require_mobile_user, sign_path
from app.storage import Storage


router = APIRouter(tags=["mobile-reports"], dependencies=[Depends(require_mobile_user)])


ReportStatus = Literal["pending", "ready", "failed"]


class Report(BaseModel):
    id: str
    studyId: str
    status: ReportStatus
    pdfUrl: Optional[str] = None
    docxUrl: Optional[str] = None
    generatedAt: Optional[str] = None


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


def _iso_utc(ts_epoch: float) -> str:
    return datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _build_report(sid: str) -> Report:
    store = _store()
    root = store.study_root(sid)
    if not root.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Study not found")

    exports = root / "exports"
    pdf: Path = exports / "report.pdf"
    docx: Path = exports / "report.docx"

    pdf_exists = pdf.exists()
    docx_exists = docx.exists()
    if not pdf_exists and not docx_exists:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No report generated yet")

    newest_ts = max(
        (pdf.stat().st_mtime if pdf_exists else 0),
        (docx.stat().st_mtime if docx_exists else 0),
    )

    return Report(
        id=f"{sid}-report",
        studyId=sid,
        status="ready",
        pdfUrl=sign_path(f"/api/mobile/v1/studies/{sid}/report/pdf") if pdf_exists else None,
        docxUrl=sign_path(f"/api/mobile/v1/studies/{sid}/report/docx") if docx_exists else None,
        generatedAt=_iso_utc(newest_ts),
    )


@router.get("/studies/{sid}/report", response_model=Report)
def get_report(sid: str) -> Report:
    return _build_report(sid)


@router.post("/studies/{sid}/report", response_model=Report, status_code=status.HTTP_202_ACCEPTED)
def generate_report(sid: str) -> Report:
    """M1 stub: report generation is driven by the existing web-side tasks
    router. Mobile calls this to check/poll; if a report already exists we
    return 200 with it. When the task-runner integration lands (M2), this
    will enqueue a regeneration job and return 202 with status=pending."""
    store = _store()
    root = store.study_root(sid)
    if not root.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Study not found")

    try:
        return _build_report(sid)
    except HTTPException as exc:
        if exc.status_code != status.HTTP_404_NOT_FOUND:
            raise
        # No report yet — the proper flow (enqueue a task) will ship with
        # the chat/task wiring in M2. For M1 we tell the client to wait.
        return Report(
            id=f"{sid}-report",
            studyId=sid,
            status="pending",
            pdfUrl=None,
            docxUrl=None,
            generatedAt=None,
        )
